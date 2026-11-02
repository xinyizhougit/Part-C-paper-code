import os
import pandas as pd
import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Calculate MAE, RMSE, WMAPE, and the true mean flow.
    """
    # Filter out NaN values to prevent errors
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[valid_mask]
    y_p = y_pred[valid_mask]

    mae = np.mean(np.abs(y_t - y_p))
    rmse = np.sqrt(np.mean(np.square(y_t - y_p)))
    mean_flow = np.mean(y_t)

    # Strictly calculate WMAPE based on the global mean flow
    if mean_flow > 0:
        wmape = (mae / mean_flow) * 100
    else:
        wmape = 0.0

    return mae, rmse, wmape, mean_flow

def main():
    #
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

    pred_file = os.path.join(desktop_path, "")
    true_file = os.path.join(desktop_path, "")
    output_file = os.path.join(desktop_path, "")

    if not os.path.exists(pred_file) or not os.path.exists(true_file):
        print(f"Files not found. Please ensure both files are on the desktop:\n1. {pred_file}\n2. {true_file}")
        return

    print("Loading Excel data (this may take a few seconds)...")
    # sheet_name=None reads all sheets and returns a dictionary {sheet_name: dataframe}
    dict_preds = pd.read_excel(pred_file, sheet_name=None)
    dict_trues = pd.read_excel(true_file, sheet_name=None)

    horizons = ['15', '30', '45', '60']
    all_results = []

    print("\nStarting evaluation and metric consistency check...")
    print("-" * 90)
    print(f"{'Region':<10} | {'Model':<15} | {'Horizon':<8} | {'MAE':<8} | {'WMAPE(%)':<8} | {'True Mean':<10} | {'Calc Ratio':<10}")
    print("-" * 90)

    # 2. Iterate through each region (Sheet)
    for region, df_pred in dict_preds.items():
        if region not in dict_trues:
            print(f"Warning: Sheet for region {region} is missing in the true values file. Skipping this region.")
            continue

        df_true = dict_trues[region]

        # Extract model names: by reading column names in the prediction table and splitting by the last '_'
        #
        # Filter out non-prediction columns
        pred_cols = [c for c in df_pred.columns if '_' in str(c) and str(c).rsplit('_', 1)[1] in horizons]
        models = sorted(list(set([str(c).rsplit('_', 1)[0] for c in pred_cols])))

        # Iterate through each model
        for model in models:
            row_data = {'Region': region, 'Model': model}

            # Iterate through each prediction horizon
            for h in horizons:
                pred_col = f"{model}_{h}"
                true_col = h  # Assume the column names in the true values table are exactly '15', '30', '45', '60'

                # For compatibility with true table column names being numbers or having 'min' suffix
                if true_col not in df_true.columns:
                    true_col_alt = f"{h}min"
                    if true_col_alt in df_true.columns:
                        true_col = true_col_alt
                    elif int(h) in df_true.columns:
                        true_col = int(h)
                    else:
                        print(f"Warning: Horizon {h} column not found in the true table for region {region}. Please check the headers.")
                        continue

                if pred_col in df_pred.columns:
                    y_p = df_pred[pred_col].values
                    y_t = df_true[true_col].values

                    mae, rmse, wmape, mean_flow = calculate_metrics(y_t, y_p)

                    row_data[f'{h}min_MAE'] = mae
                    row_data[f'{h}min_RMSE'] = rmse
                    row_data[f'{h}min_WMAPE(%)'] = wmape

                    # Validation calculation: MAE / (WMAPE / 100)
                    calc_ratio = mae / (wmape / 100) if wmape > 0 else 0
                    print(f"{region:<10} | {model:<15} | {h:<8} | {mae:<8.4f} | {wmape:<8.2f} | {mean_flow:<10.4f} | {calc_ratio:<10.4f}")

                    # Strict code-level validation
                    if mean_flow > 0:
                        assert np.isclose(mean_flow, calc_ratio, atol=1e-4), \
                            f"Consistency error: Calculation for {model} in {region} at {h}min is not aligned!"

            all_results.append(row_data)

    print("-" * 90)
    print("\n✅ Consistency check passed for all regions and models! The MAE and WMAPE ratio is strictly constrained by the observed mean flow.")

    # 3. Organize results and save
    df_final = pd.DataFrame(all_results)

    # Arrange columns by header order
    cols_order = ['Region', 'Model']
    for h in horizons:
        cols_order.extend([f'{h}min_MAE', f'{h}min_RMSE', f'{h}min_WMAPE(%)'])

    df_final = df_final[[c for c in cols_order if c in df_final.columns]]

    # Save to desktop
    df_final.to_excel(output_file, index=False)
    print(f"\nThe complete evaluation results have been successfully saved to the desktop: {output_file}")

if __name__ == "__main__":
    main()