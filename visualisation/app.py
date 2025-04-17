from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import uuid
import io
import base64
from werkzeug.utils import secure_filename
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

app = Flask(__name__)
app.secret_key = "your_secret_key_here"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls', 'tsv'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/upload', methods=['POST'])

def upload_file():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Generate unique ID for this session
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(file_path)
        session['file_path'] = file_path
        
        # Load data
        try:
            file_ext = filename.rsplit('.', 1)[1].lower()
            if file_ext == 'csv':
                df = pd.read_csv(file_path)
            elif file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
            elif file_ext == 'tsv':
                df = pd.read_csv(file_path, sep='\t')
            
            # Basic data info
            session['columns'] = df.columns.tolist()
            session['dtypes'] = df.dtypes.astype(str).to_dict()
            session['rows'] = len(df)
            
            # Identify numeric and categorical columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            
            session['numeric_cols'] = numeric_cols
            session['categorical_cols'] = categorical_cols
            
            # Generate basic EDA information
            eda_info = {}
            
            # Numeric columns stats
            for col in numeric_cols:
                eda_info[col] = {
                    'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    'median': float(df[col].median()) if not pd.isna(df[col].median()) else None,
                    'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
                    'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    'null_count': int(df[col].isna().sum())
                }
            
            # Categorical columns stats
            for col in categorical_cols:
                value_counts = df[col].value_counts().head(10).to_dict()  # Top 10 values
                eda_info[col] = {
                    'unique_count': int(df[col].nunique()),
                    'top_values': {str(k): int(v) for k, v in value_counts.items()},
                    'null_count': int(df[col].isna().sum())
                }
            
            # Store EDA info in session as JSON string
            session['eda_info'] = json.dumps(eda_info)
            
            flash('File uploaded successfully!', 'success')
            return redirect(url_for('preprocess'))
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            if os.path.exists(file_path):
                os.remove(file_path)
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload CSV, Excel, or TSV files.', 'error')
    return redirect(url_for('index'))

# @app.route('/preprocess', methods=['GET', 'POST'])
# def preprocess():
#     if 'file_path' not in session:
#         flash('Please upload a file first', 'error')
#         return redirect(url_for('index'))
    
#     if request.method == 'POST':
#         # Get the file
#         file_path = session['file_path']
        
#         # Load the dataframe
#         file_ext = file_path.rsplit('.', 1)[1].lower()
#         if file_ext == 'csv':
#             df = pd.read_csv(file_path)
#         elif file_ext in ['xlsx', 'xls']:
#             df = pd.read_excel(file_path)
#         elif file_ext == 'tsv':
#             df = pd.read_csv(file_path, sep='\t')
        
#         # Handle missing values
#         missing_strategy = request.form.get('missing_strategy', 'none')
        
#         if missing_strategy == 'drop_rows':
#             df = df.dropna()
#         elif missing_strategy == 'drop_columns':
#             df = df.dropna(axis=1)
#         elif missing_strategy == 'fill_mean':
#             for col in df.select_dtypes(include=['int64', 'float64']).columns:
#                 df[col] = df[col].fillna(df[col].mean())
#         elif missing_strategy == 'fill_median':
#             for col in df.select_dtypes(include=['int64', 'float64']).columns:
#                 df[col] = df[col].fillna(df[col].median())
#         elif missing_strategy == 'fill_mode':
#             for col in df.columns:
#                 df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else df[col])
#         elif missing_strategy == 'fill_zero':
#             df = df.fillna(0)
        
#         # Scaling for numeric columns
#         scaling_strategy = request.form.get('scaling_strategy', 'none')
#         if scaling_strategy != 'none':
#             for col in df.select_dtypes(include=['int64', 'float64']).columns:
#                 if scaling_strategy == 'standard':
#                     scaler = StandardScaler()
#                     df[col] = scaler.fit_transform(df[[col]])
#                 elif scaling_strategy == 'minmax':
#                     scaler = MinMaxScaler()
#                     df[col] = scaler.fit_transform(df[[col]])
        
#         # Encoding for categorical columns
#         encoding_strategy = request.form.get('encoding_strategy', 'none')
#         if encoding_strategy != 'none':
#             for col in df.select_dtypes(include=['object', 'category']).columns:
#                 if encoding_strategy == 'label':
#                     le = LabelEncoder()
#                     df[col] = le.fit_transform(df[col].astype(str))
#                 elif encoding_strategy == 'onehot':
#                     # Only use one-hot encoding for columns with fewer than 15 unique values to avoid explosion
#                     if df[col].nunique() < 15:
#                         dummies = pd.get_dummies(df[col], prefix=col)
#                         df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
        
#         # Save the processed dataframe
#         processed_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session['session_id']}_processed.csv")
#         df.to_csv(processed_file_path, index=False)
#         session['processed_file_path'] = processed_file_path
        
#         # Update column information after processing
#         session['columns'] = df.columns.tolist()
#         session['dtypes'] = df.dtypes.astype(str).to_dict()
#         session['rows'] = len(df)
        
#         # Update numeric and categorical columns
#         numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
#         categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
#         session['numeric_cols'] = numeric_cols
#         session['categorical_cols'] = categorical_cols
        
#         flash('Data preprocessing completed!', 'success')
#         return redirect(url_for('visualization'))
    
#     # For GET requests, render the preprocessing form
#     return render_template('preprocess.html', 
#                           columns=session.get('columns', []),
#                           numeric_cols=session.get('numeric_cols', []),
#                           categorical_cols=session.get('categorical_cols', []),
#                           rows=session.get('rows', 0),
#                           eda_info=json.loads(session.get('eda_info', '{}')))

@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    if 'file_path' not in session:
        flash('Please upload a file first', 'error')
        return redirect(url_for('index'))
    
    file_path = session['file_path']
    file_ext = file_path.rsplit('.', 1)[1].lower()

    # Reload data from the file to reflect modifications
    try:
        if file_ext == 'csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(file_path)
        elif file_ext == 'tsv':
            df = pd.read_csv(file_path, sep='\t')
    except Exception as e:
        flash(f'Error loading updated file: {str(e)}', 'error')
        return redirect(url_for('index'))

    # Update session details dynamically
    session['columns'] = df.columns.tolist()
    session['dtypes'] = df.dtypes.astype(str).to_dict()
    session['rows'] = len(df)

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    session['numeric_cols'] = numeric_cols
    session['categorical_cols'] = categorical_cols

    # Generate EDA info dynamically
    eda_info = {}

    # Numeric column statistics
    for col in numeric_cols:
        eda_info[col] = {
            'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
            'median': float(df[col].median()) if not pd.isna(df[col].median()) else None,
            'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
            'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
            'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
            'null_count': int(df[col].isna().sum())
        }

    # Categorical column statistics
    for col in categorical_cols:
        value_counts = df[col].value_counts().head(10).to_dict()
        eda_info[col] = {
            'unique_count': int(df[col].nunique()),
            'top_values': {str(k): int(v) for k, v in value_counts.items()},
            'null_count': int(df[col].isna().sum())
        }

    # Store EDA info in session
    session['eda_info'] = json.dumps(eda_info)

    if request.method == 'POST':
        # Handle missing values
        missing_strategy = request.form.get('missing_strategy', 'none')

        if missing_strategy == 'drop_rows':
            df = df.dropna()
        elif missing_strategy == 'drop_columns':
            df = df.dropna(axis=1)
        elif missing_strategy == 'fill_mean':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())
        elif missing_strategy == 'fill_median':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
        elif missing_strategy == 'fill_mode':
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else df[col])
        elif missing_strategy == 'fill_zero':
            df = df.fillna(0)

        # Scaling for numeric columns
        scaling_strategy = request.form.get('scaling_strategy', 'none')
        if scaling_strategy != 'none':
            for col in numeric_cols:
                if scaling_strategy == 'standard':
                    scaler = StandardScaler()
                    df[col] = scaler.fit_transform(df[[col]])
                elif scaling_strategy == 'minmax':
                    scaler = MinMaxScaler()
                    df[col] = scaler.fit_transform(df[[col]])

        # Encoding for categorical columns
        encoding_strategy = request.form.get('encoding_strategy', 'none')
        if encoding_strategy != 'none':
            for col in categorical_cols:
                if encoding_strategy == 'label':
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                elif encoding_strategy == 'onehot' and df[col].nunique() < 15:
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df.drop(col, axis=1), dummies], axis=1)

        # Save the processed file
        processed_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session['session_id']}_processed.csv")
        df.to_csv(processed_file_path, index=False)
        session['processed_file_path'] = processed_file_path

        # Update session details again
        session['columns'] = df.columns.tolist()
        session['dtypes'] = df.dtypes.astype(str).to_dict()
        session['rows'] = len(df)

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        session['numeric_cols'] = numeric_cols
        session['categorical_cols'] = categorical_cols

        flash('Data preprocessing completed!', 'success')
        return redirect(url_for('visualization'))

    return render_template('preprocess.html', 
                           columns=session.get('columns', []),
                           numeric_cols=session.get('numeric_cols', []),
                           categorical_cols=session.get('categorical_cols', []),
                           rows=session.get('rows', 0),
                           eda_info=json.loads(session.get('eda_info', '{}')))

@app.route('/visualization', methods=['GET', 'POST'])
def visualization():
    if 'file_path' not in session:
        flash('Please upload a file first', 'error')
        return redirect(url_for('index'))
    
    # Determine which file to use (processed or original)
    file_to_use = session.get('processed_file_path', session['file_path'])
    
    # Load the dataframe
    file_ext = file_to_use.rsplit('.', 1)[1].lower()
    if file_ext == 'csv':
        df = pd.read_csv(file_to_use)
    elif file_ext in ['xlsx', 'xls']:
        df = pd.read_excel(file_to_use)
    elif file_ext == 'tsv':
        df = pd.read_csv(file_to_use, sep='\t')
    
    # If POST, generate visualization
    if request.method == 'POST':
        # Get form data
        plot_type = request.form.get('plot_type')
        x_column = request.form.get('x_column')
        y_column = request.form.get('y_column')
        hue_column = request.form.get('hue_column', None)
        
        if hue_column == 'none':
            hue_column = None
            
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Generate plot based on type
        if plot_type == 'scatter':
            sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue_column)
            plt.title(f'Scatter Plot: {y_column} vs {x_column}')
            
        elif plot_type == 'line':
            sns.lineplot(data=df, x=x_column, y=y_column, hue=hue_column)
            plt.title(f'Line Plot: {y_column} vs {x_column}')
            
        elif plot_type == 'bar':
            sns.barplot(data=df, x=x_column, y=y_column, hue=hue_column)
            plt.title(f'Bar Plot: {y_column} by {x_column}')
            plt.xticks(rotation=45)
            
        elif plot_type == 'hist':
            sns.histplot(data=df, x=x_column, kde=True)
            plt.title(f'Histogram of {x_column}')
            
        elif plot_type == 'box':
            sns.boxplot(data=df, x=x_column, y=y_column, hue=hue_column)
            plt.title(f'Box Plot: {y_column} by {x_column}')
            plt.xticks(rotation=45)
            
        elif plot_type == 'violin':
            sns.violinplot(data=df, x=x_column, y=y_column, hue=hue_column)
            plt.title(f'Violin Plot: {y_column} by {x_column}')
            plt.xticks(rotation=45)
            
        elif plot_type == 'count':
            sns.countplot(data=df, x=x_column, hue=hue_column)
            plt.title(f'Count Plot of {x_column}')
            plt.xticks(rotation=45)
            
        elif plot_type == 'heatmap':
            if x_column == y_column:
                corr_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
                plt.title('Correlation Heatmap')
            else:
                pivot_table = pd.pivot_table(df, values=y_column, index=x_column, columns=hue_column)
                sns.heatmap(pivot_table, annot=True, cmap='coolwarm')
                plt.title(f'Heatmap of {y_column} by {x_column} and {hue_column}')
            
        elif plot_type == 'pie':
            print('x_column:', x_column)
            # Calculate value counts
            counts = df[x_column].value_counts()
            # Only show top 10 categories if there are too many
            if len(counts) > 10:
                other_count = counts[10:].sum()
                counts = counts[:10]
                counts['Other'] = other_count
            plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
            plt.title(f'Pie Chart of {x_column}')
            
        elif plot_type == 'density':
            sns.kdeplot(data=df, x=x_column, hue=hue_column)
            plt.title(f'Density Plot of {x_column}')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        
        # Encode the plot to base64 for embedding in HTML
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        
        plt.close()  # Close the figure to free memory
        
        return render_template('visualization.html',
                              columns=session.get('columns', []),
                              numeric_cols=session.get('numeric_cols', []),
                              categorical_cols=session.get('categorical_cols', []),
                              rows=session.get('rows', 0),
                              plot_url=plot_url,
                              plot_type=plot_type,
                              x_column=x_column,
                              y_column=y_column,
                              hue_column=hue_column)
    
    # For GET requests, render the visualization form
    return render_template('visualization.html',
                          columns=session.get('columns', []),
                          numeric_cols=session.get('numeric_cols', []),
                          categorical_cols=session.get('categorical_cols', []),
                          rows=session.get('rows', 0))

@app.route('/data_sample', methods=['GET'])
def data_sample():
    if 'file_path' not in session:
        return jsonify({'error': 'No file uploaded'})
    
    # Determine which file to use (processed or original)
    file_to_use = session.get('processed_file_path', session['file_path'])
    
    # Load the dataframe
    file_ext = file_to_use.rsplit('.', 1)[1].lower()
    if file_ext == 'csv':
        df = pd.read_csv(file_to_use)
    elif file_ext in ['xlsx', 'xls']:
        df = pd.read_excel(file_to_use)
    elif file_ext == 'tsv':
        df = pd.read_csv(file_to_use, sep='\t')
    
    # Return a sample of the data (first 10 rows)
    return jsonify({
        'columns': df.columns.tolist(),
        'data': df.head(10).replace({np.nan: None}).to_dict('records')
    })

@app.route('/download_processed', methods=['GET'])
def download_processed():
    if 'processed_file_path' not in session:
        flash('No processed file available', 'error')
        return redirect(url_for('preprocess'))
    
    # Send the processed file
    return send_file(session['processed_file_path'], as_attachment=True, download_name='processed_data.csv')

@app.route('/reset', methods=['GET'])
def reset():
    # Remove files associated with the session
    if 'file_path' in session and os.path.exists(session['file_path']):
        os.remove(session['file_path'])
    
    if 'processed_file_path' in session and os.path.exists(session['processed_file_path']):
        os.remove(session['processed_file_path'])
    
    # Clear session
    session.clear()
    
    flash('Session reset. You can upload a new file.', 'info')
    return redirect(url_for('index'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)