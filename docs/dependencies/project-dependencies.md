# Project dependencies

- **[pandas](https://pandas.pydata.org/docs/):** Reads healthcare CSV datasets and extracts, cleans, and normalizes member data for the ETL pipeline.
- **[scikit-learn](https://scikit-learn.org/stable/):** Prepares model inputs (filling blanks, scaling, encoding text), trains and evaluates the Random Forest risk model, and (via its bundled **joblib**) saves the fitted model to disk.
- **[XGBoost](https://xgboost.readthedocs.io/):** The gradient-boosted tree model, compared against Random Forest and the majority baseline on the same split and metrics.
- **[Streamlit](https://docs.streamlit.io/):** Provides a lightweight web interface to display member records and relevant features from the SQL database.
