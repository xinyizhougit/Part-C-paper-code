from parameters import parse_config
from data_processing import prepare


def run(c, action):
    if action == 'paired':
        from experiments import paired_days
        if not c.baseline_predictions or not c.checkpoint:
            raise ValueError('paired uses --checkpoint for ours test_predictions.npz and --baseline-predictions for baseline npz')
        print(paired_days(c.checkpoint, c.baseline_predictions, c.output))
        return
    data = prepare(c)
    print(f"Prepared: {len(data['nodes'])} nodes, {len(data['x'])} time bins", flush=True)
    if action == 'prepare':
        from data_audit import audit_csv
        from pathlib import Path
        audit_csv(c.data, Path(c.output, 'all_region_audit.json'))
        return
    import json
    from pathlib import Path
    report = json.loads(Path(c.output, 'data_report.json').read_text())
    if c.mode == 'strict' and not report['coverage']['covers_requested_span']:
        raise ValueError('Strict case-study run requires complete September-October coverage. '
                         'Current CSV is partial; inspect data_report.json or explicitly use exploratory mode.')
    from graph_construction import GraphBuilder
    builder = GraphBuilder(data, c)
    builder.calibrate()
    import numpy as np
    from pathlib import Path
    np.savez_compressed(Path(c.output, 'frozen_graph_priors.npz'), physical_adjacency=builder.st,
                        road_km=builder.road, pearson=builder.rho, attractiveness=builder.attract)
    from training import train, grid_search, repeat_runs
    if action == 'grid':
        grid_search(data, builder, c)
    elif action == 'repeat':
        repeat_runs(data, builder, c)
    elif action == 'benchmark':
        from experiments import benchmark
        print(benchmark(data, builder, c))
    else:
        train(data, builder, c)


def main():
    from dataclasses import replace
    from pathlib import Path
    c, action = parse_config()
    if c.region == 'ALL':
        regions = ['MRCLM', 'OMITSJ', 'CTLSV', 'MRCLASSO', 'LRRS', 'CITVR', 'CITCRC', 'CITLA', 'CITSO', 'CITPI', 'CITHSL']
        for region in regions:
            cc = replace(c, region=region, output=str(Path(c.output, region)),
                         external=c.external.format(region=region) if c.external else '')
            Path(cc.output).mkdir(parents=True, exist_ok=True)
            run(cc, action)
    else:
        run(c, action)


if __name__ == '__main__':
    main()
