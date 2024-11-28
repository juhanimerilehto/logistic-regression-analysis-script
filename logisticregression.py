import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def perform_logistic_regression(excel_path='data.xlsx',
                              feature_columns=['Feature1', 'Feature2', 'Feature3'],
                              target_column='Target',
                              test_size=0.2,
                              random_state=42,
                              output_prefix='logistic'):
    """
    Performs logistic regression analysis with visualization and model evaluation.
    
    Parameters:
    -----------
    excel_path : str
        Path to Excel file containing the data
    feature_columns : list
        List of column names to use as features
    target_column : str
        Name of the column containing the binary target variable
    test_size : float
        Proportion of data to use for testing (0.0 to 1.0)
    random_state : int
        Random seed for reproducibility
    output_prefix : str
        Prefix for output files
    """
    
    # Read the data
    print(f"Reading data from {excel_path}...")
    df = pd.read_excel(excel_path)
    
    # Prepare features and target
    X = df[feature_columns]
    y = df[target_column]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    # Fit model
    model = LogisticRegression(random_state=random_state)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Create coefficient summary
    coef_summary = pd.DataFrame({
        'Feature': feature_columns,
        'Coefficient': model.coef_[0],
        'Abs_Coefficient': abs(model.coef_[0])
    }).sort_values('Abs_Coefficient', ascending=False)
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save results to Excel
    excel_output = f'{output_prefix}_results_{timestamp}.xlsx'
    with pd.ExcelWriter(excel_output) as writer:
        # Save model coefficients
        coef_summary.to_excel(writer, sheet_name='Coefficients', index=False)
        
        # Save classification report
        pd.DataFrame(class_report).transpose().to_excel(
            writer, sheet_name='Classification Report'
        )
        
        # Save confusion matrix
        pd.DataFrame(
            conf_matrix,
            columns=['Predicted 0', 'Predicted 1'],
            index=['Actual 0', 'Actual 1']
        ).to_excel(writer, sheet_name='Confusion Matrix')
        
        # Save test set predictions
        pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Probability': y_pred_prob
        }).to_excel(writer, sheet_name='Predictions', index=False)
    
    print("\nResults saved to:", excel_output)
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confusion Matrix Heatmap
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix')
    
    # 2. Feature Importance Plot
    coef_summary.plot(kind='bar', x='Feature', y='Abs_Coefficient', ax=ax2)
    ax2.set_title('Feature Importance')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    # 3. Probability Distribution
    sns.histplot(data=pd.DataFrame({
        'Probability': y_pred_prob,
        'Actual': y_test
    }), x='Probability', hue='Actual', bins=20, ax=ax3)
    ax3.set_title('Prediction Probability Distribution')
    
    # 4. ROC Curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    ax4.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax4.plot([0, 1], [0, 1], 'k--')
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curve')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_output = f'{output_prefix}_plot_{timestamp}.png'
    plt.savefig(plot_output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plot saved to:", plot_output)
    
    # Print results to terminal
    print("\nLogistic Regression Results:")
    print("---------------------------")
    print("\nModel Coefficients:")
    print(coef_summary)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\nConfusion Matrix:")
    print(conf_matrix)

if __name__ == "__main__":
    # Example usage:
    # Modify these parameters according to your data
    perform_logistic_regression(
        excel_path='data.xlsx',
        feature_columns=['Feature1', 'Feature2', 'Feature3'],
        target_column='Target',
        test_size=0.2,
        random_state=42,
        output_prefix='logistic'
    )