import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
PATH_TO_FILE = '../datasets/City_Temperature.csv'


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    
    df = pd.read_csv(filename, parse_dates=["Date"])
    df = df[df['Temp'] > -45]
    df = df.dropna().drop_duplicates()
    df["DM"] = df["Date"].dt.strftime('%d-%m')
    df["Year"] = df["Year"].astype(str)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    return df


def israel_data_analysis(df):
    israel_data = df[df.Country == "Israel"]

    px.scatter(israel_data, x="DayOfYear", y="Temp", color="Year")\
        .write_image("israel_tmp_vs_dayOfYear.png")

    monthly_std = israel_data.groupby(["Month"], as_index=False).agg(std=("Temp", "std"))

    px.bar(monthly_std, title="monthly temp Standard Deviation in israel",
                                   x="Month", y="std").write_image("monthly_std_israel.png")
    return israel_data

def monthly_temp_by_country(df):
    groupby_month_country = df.groupby(["Month", "Country"], as_index=False)
    aggregated = groupby_month_country.agg(mean=("Temp", "mean"),
                                                std=("Temp", "std"))
    line_plot = px.line(aggregated, x="Month", y="mean", error_y="std",
                        color="Country")
    line_plot.update_layout(title="Mean Monthly Temperatures",
                            xaxis_title="Month",
                            yaxis_title="Average Temperature")
    line_plot.write_image("Monthly_temperatures_vs_country_Average.png")

def polinomial_regression_on_israel_data(df):
    train_X, train_Y, test_X, test_Y = split_train_test(df.DayOfYear, df.Temp)
    np_train_x = train_X.to_numpy()
    np_train_y = train_Y.to_numpy()
    np_test_x = test_X.to_numpy()
    np_test_y = test_Y.to_numpy()

    polynomial_degrees = [i for i in range(1, 11)]
    errs = np.zeros_like(polynomial_degrees, dtype=float)

    for i, j in enumerate(polynomial_degrees):
        polynomial = PolynomialFitting(k=j)
        model = polynomial.fit(np_train_x, np_train_y)
        errs[i] = np.round(model.loss(np_test_x, np_test_y), 2)

    errs_df = pd.DataFrame(dict(degree=polynomial_degrees, error=errs))

    px.bar(errs_df, x="degree", y="error", text="error",
           title="Errors of polynomial models with different degrees")\
        .write_image("Errors of polynomial models with different degrees.png")
    print(errs_df)


def non_israel_countries_fit(df, df_israel, deg):
    np_israel_day = df_israel.DayOfYear.to_numpy()
    np_israel_temp = df_israel.Temp.to_numpy()
    poly_fit = PolynomialFitting(k=deg).fit(np_israel_day, np_israel_temp)
    countries = df.Country.unique()
    countries = [country for country in countries if country != 'Israel']
    country_errors = []
    for country in countries:
        country_data = df[df.Country == country]
        loss = round(poly_fit.loss(country_data.DayOfYear, country_data.Temp), 2)
        error_dict = {"country": country, "error": loss}
        country_errors.append(error_dict)
    errors_df = pd.DataFrame(country_errors)
    error_bar_plot = px.bar(errors_df, x="country", y="error", text="error",
                            color="country",
                            title="Error for countries by model fitted on Israel")

    error_bar_plot.write_image("Error for countries by model fitted on Israel.png")


if __name__ == '__main__':
    np.random.seed(0)
    data = load_data(PATH_TO_FILE)

    israel_df = israel_data_analysis(data)
    monthly_temp_by_country(data)
    polinomial_regression_on_israel_data(israel_df)
    non_israel_countries_fit(data, israel_df, 5)