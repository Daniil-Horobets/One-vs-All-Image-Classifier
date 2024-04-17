import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math


'''
    .. Use this script to plot and analyze the training process of a model.
        It will plot 3 graphs:
            - Training Loss and Validation Loss Over Epochs
            - Train Metrics Over Epochs
            - Validation Metrics Over Epochs
        Next metrics are plotted:
            ['Sensitivity', 'Specificity', 'Precision', 'Negative Predictive Value', 'Accuracy', 'F1 Score',
            'Matthews Correlation Coefficient', 'False Positive Rate', 'False Discovery Rate', 'False Negative Rate']
'''


def calculate_metrics(TN, FP, FN, TP):
    epsilon = 1e-15
    sensitivity = TP / (TP + FN + epsilon)
    specificity = TN / (FP + TN + epsilon)
    precision = TP / (TP + FP + epsilon)
    npv = TN / (TN + FN + epsilon)
    fpr = FP / (FP + TN + epsilon)
    fdr = FP / (FP + TP + epsilon)
    fnr = FN / (FN + TP + epsilon)
    accuracy = (TP + TN) / (TP + FP + FN + TN + epsilon)
    f1_score = 2 * TP / (2 * TP + FP + FN + epsilon)
    mcc = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + epsilon)

    return {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "Negative Predictive Value": npv,
        "False Positive Rate": fpr,
        "False Discovery Rate": fdr,
        "False Negative Rate": fnr,
        "Accuracy": accuracy,
        "F1 Score": f1_score,
        "Matthews Correlation Coefficient": mcc
    }


def analyze_single(log_file_path, model_type):

    df = pd.read_csv(log_file_path, delimiter=';')
    df['epoch'] += 1


    # Plot training_loss and validation_loss without dots using Plotly Express
    fig = px.line(df, x='epoch', y=[f'train_loss_{model_type}', f'val_loss_{model_type}'],
                  labels={'value': 'Loss', 'variable': 'Metric'},
                  title='Training Loss and Validation Loss Over Epochs')
    # Add annotations to data points using hover text
    fig.update_traces(hoverinfo='text',
                      text=[f'Epoch: {epoch}\nLoss: {loss:.6f}' for epoch, loss in
                            zip(df['epoch'], df[f'train_loss_{model_type}'])])
    train_columns = [f'train_TN_{model_type}', f'train_FP_{model_type}', f'train_FN_{model_type}', f'train_TP_{model_type}']
    val_columns = [f'val_TN_{model_type}', f'val_FP_{model_type}', f'val_FN_{model_type}', f'val_TP_{model_type}']
    # Create new DataFrames for train and val metrics vs epochs
    train_metrics_df = pd.DataFrame(columns=["epoch"] + list(calculate_metrics(0, 0, 0, 0).keys()))
    val_metrics_df = pd.DataFrame(columns=["epoch"] + list(calculate_metrics(0, 0, 0, 0).keys()))
    # Iterate through each row in the original DataFrame and calculate metrics
    for index, row in df.iterrows():
        epoch = row['epoch']
        train_metrics = calculate_metrics(row[train_columns[0]], row[train_columns[1]], row[train_columns[2]],
                                          row[train_columns[3]])
        val_metrics = calculate_metrics(row[val_columns[0]], row[val_columns[1]], row[val_columns[2]],
                                        row[val_columns[3]])

        # Append metrics to the new DataFrames
        train_metrics_df = pd.concat([train_metrics_df, pd.DataFrame({"epoch": [epoch], **train_metrics})],
                                     ignore_index=True)
        val_metrics_df = pd.concat([val_metrics_df, pd.DataFrame({"epoch": [epoch], **val_metrics})], ignore_index=True)
    fig2 = px.line(train_metrics_df, x='epoch',
                   y=['Sensitivity', 'Specificity', 'Precision', 'Negative Predictive Value',
                      'Accuracy', 'Matthews Correlation Coefficient'],
                   labels={'value': 'Gain', 'variable': 'Metric'},
                   title='Train Metrics Over Epochs')
    fig2.add_trace(
        go.Scatter(x=train_metrics_df['epoch'], y=train_metrics_df['F1 Score'], mode='lines', line=dict(width=4),
                   name='F1 Score'))
    fig2.add_trace(go.Scatter(x=train_metrics_df['epoch'], y=train_metrics_df['False Positive Rate'], mode='lines',
                              line=dict(width=1), name='False Positive Rate'))
    fig2.add_trace(go.Scatter(x=train_metrics_df['epoch'], y=train_metrics_df['False Discovery Rate'], mode='lines',
                              line=dict(width=1), name='False Discovery Rate'))
    fig2.add_trace(go.Scatter(x=train_metrics_df['epoch'], y=train_metrics_df['False Negative Rate'], mode='lines',
                              line=dict(width=1), name='False Negative Rate'))
    fig3 = px.line(val_metrics_df, x='epoch', y=['Sensitivity', 'Specificity', 'Precision', 'Negative Predictive Value',
                                                 'Accuracy', 'Matthews Correlation Coefficient'],
                   labels={'value': 'Gain', 'variable': 'Metric'},
                   title='Validation Metrics Over Epochs')
    fig3.add_trace(go.Scatter(x=val_metrics_df['epoch'], y=val_metrics_df['F1 Score'], mode='lines', line=dict(width=4),
                              name='F1 Score'))
    fig3.add_trace(
        go.Scatter(x=val_metrics_df['epoch'], y=val_metrics_df['False Positive Rate'], mode='lines', line=dict(width=1),
                   name='False Positive Rate'))
    fig3.add_trace(go.Scatter(x=val_metrics_df['epoch'], y=val_metrics_df['False Discovery Rate'], mode='lines',
                              line=dict(width=1), name='False Discovery Rate'))
    fig3.add_trace(
        go.Scatter(x=val_metrics_df['epoch'], y=val_metrics_df['False Negative Rate'], mode='lines', line=dict(width=1),
                   name='False Negative Rate'))
    # Display the figures
    fig.show()
    fig2.show()
    fig3.show()

if __name__ == '__main__':

    analyze_single(log_file_path='log_file_crack.csv', model_type='crack')

    # analyze_single(log_file_path='log_file_inactive.csv', model_type='inactive')

