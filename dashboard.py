import pandas as pd
import sqlite3
import plotly.express as px
import streamlit as st
import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu
from pathlib import Path
import yaml
from yaml.loader import SafeLoader
import smtplib


def send_email(TO,subject,body):
    # EMAIL_ADDRESS=os.getenv('MAIL_USER')
    # EMAIL_PASSWORD = os.environ.get('MAIL_PASS')
    EMAIL_ADDRESS=st.secrets['MAIL_USER']
    EMAIL_PASSWORD = st.secrets['MAIL_PASS']
    host = 'smtp.gmail.com'
    port = 587
    with smtplib.SMTP(host,port) as smtp:
        smtp.starttls()
        smtp.login(EMAIL_ADDRESS,EMAIL_PASSWORD)
        FROM = EMAIL_PASSWORD
        msg = f'Subject:{subject}\n\n{body}'
        smtp.sendmail(FROM, TO, msg)
    
def format_number(value, decimal_place=2, pos_sign=False):
    if abs(value) < 1e3:
        formatted_string = str(value)
    elif abs(value) < 1e6:
        formatted_string = f"{value / 1e3:.{decimal_place}f}k"
    elif abs(value) < 1e9:
        formatted_string = f"{value / 1e6:.{decimal_place}f}M"
    elif abs(value) < 1e12:
        formatted_string = f"{value / 1e9:.{decimal_place}f}B"
    else:
        formatted_string = f"{value / 1e12:.{decimal_place}f}T"

    if value >= 0 and pos_sign:
        return f"+{formatted_string}"
    else:
        return formatted_string


def format_percentage(value, decimal_place=2, pos_sign=False):
    if isinstance(value, (int, float)):
        percentage_value = str(round(value * 100, decimal_place))

        if value > 0 and pos_sign:
            return f"+{percentage_value}%"
        else:
            return f"{percentage_value}%"


def format_percentage_point(value, decimal_place=2, pos_sign=False):
    if isinstance(value, (int, float)):
        point_string = str(round(value * 100, decimal_place))

        if value > 0 and pos_sign:
            return f"+{point_string} pp"
        else:
            return f"{point_string} pp"

@st.cache_data
def get_data(): 
    # get data
    df = pd.read_csv("MOCK_DATA_01.csv", index_col=0)
    # transform data
    df = df[df['product_category'].isin(['Trade Finance','Corporate Bond','Loan'])]
    df["datadate"] = pd.to_datetime(df["datadate"], format="%Y-%m-%d")
    df["total_exposure_aud"] = (
        df["on_bs_amount_aud"] + df["off_bs_amount_aud"] + df["undrawn_amount_aud"]
    )
    df["off_bs_exposure_aud"] = df["off_bs_amount_aud"] + df["undrawn_amount_aud"]
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["days_to_maturity"] = (df["end_date"] - df["start_date"]).dt.days
    df["term_days"] = (df["end_date"] - df["datadate"]).dt.days
    df["maturity_category"] = pd.cut(
        df["days_to_maturity"],
        bins=[-1, 30, 90, 180, 365, 3 * 365, 5 * 365, float("inf")],
        labels=["<1mo", "1~3mo", "3-6mo", "6mo-1yr", "1-3yr", "3-5yr", ">5yr"],
    )
    df["term_category"] = pd.cut(
        df["term_days"],
        bins=[-1, 30, 90, 180, 365, 3 * 365, 5 * 365, float("inf")],
        labels=["<1mo", "1~3mo", "3-6mo", "6mo-1yr", "1-3yr", "3-5yr", ">5yr"],
    )
    df["state"] = df["state"].fillna("Unknown")
    return df


def filter_data_except_date(
    df,filt_clientName,filt_productCate,filt_productType,\
    filt_anzsicDiv,filt_anzsicSubdiv,filt_anzsicGroup,filt_hoSector,filt_cptSector,\
    filt_watchlist,filt_termDays,filt_days2maturity, filt_ppp,filt_cre,filt_rp):
    df_filtered = df[
        (
            df["client_name"].str.contains(filt_clientName, case=False)
            if filt_clientName
            else True
        )
        & (df["product_category"].isin(filt_productCate) if filt_productCate else True)
        & (df["product_type"].isin(filt_productType) if filt_productType else True)
        & (df["anzsic_division"].isin(filt_anzsicDiv) if filt_anzsicDiv else True)
        & (
            df["anzsic_subdivision"].isin(filt_anzsicSubdiv)
            if filt_anzsicSubdiv
            else True
        )
        & (df["anzsic_group"].isin(filt_anzsicGroup) if filt_anzsicGroup else True)
        & (
            df["ho_industry_classification"].isin(filt_hoSector)
            if filt_hoSector
            else True
        )
        & (
            df["cpt_industry_classification"].isin(filt_hoSector)
            if filt_hoSector
            else True
        )
        & (df["watchlist_stage"].isin(filt_watchlist) if filt_watchlist else True)
        & (
            df["days_to_maturity"].between(filt_days2maturity[0], filt_days2maturity[1])
            if filt_days2maturity
            else True
        )
        & (
            df["term_days"].between(filt_termDays[0], filt_termDays[1])
            if filt_termDays
            else True
        )
        & (~df["transaction_no"].str.startswith("RF") if filt_rp else True)
        & (df["ppp_category"].notna() if filt_ppp else True)
        & (df["cre_category"].notna() if filt_cre else True)
    ]
    return df_filtered


def generate_metric(
    st,
    label,
    value,
    compare_value = None,
    compare_method = None,
    delta_suffix = None,
    value_format="number",
    value_prefix =None,
    help = ''
):
    if value_format == "number":
        if value_prefix:
            formatted_value = f'{value_prefix}{format_number(value)}'
        else:
            formatted_value = format_number(value)
    elif value_format == "percentage":
        formatted_value = format_percentage(value)
    elif value_format in ["pp", "percentage_point"]:
        formatted_value = format_percentage_point(value)
    if compare_value != None:
        try:
            if compare_method in ["%", "percentage"]:
                delta = value / compare_value - 1
                formatted_delta = format_percentage(delta, pos_sign=True)
            elif compare_method in ["abs", "absolute"]:
                delta = value - compare_value
                formatted_delta = format_number(delta, pos_sign=True)
            elif compare_method in ["pp", "percentage_point"]:
                delta = value - compare_value
                formatted_delta = format_percentage_point(delta, pos_sign=True)

        except:
            return st.metric(label=label, value = formatted_value, help = help)
        else:
            formatted_delta_suffix = " ".join([formatted_delta, delta_suffix])
            return st.metric(label=label, value = formatted_value, delta =formatted_delta_suffix, help=help)
    else:
        return st.metric(label=label, value = formatted_value, help = help)



def generate_metrics(df_filtered,selected_date):

    st.subheader("METRICS", help="Metrics provide a quick snapshot of the branch's credit portfolio at any given month")
    col1, col2, col3 = st.columns(3, gap="small")
    delta_type = col1.selectbox(
        "Metric delta type",
        [
            "MoM % Change",
            "MoM Absolute Change",
            "YoY % Change",
            "YoY Absolute Change",
            "YTD % Change",
            "YTD Absolute Change",
        ],
    )
    col1, col2, col3, col4 = st.columns(spec=4, gap="large")

    if delta_type[:3] == "MoM":
        compareDate = selected_date + pd.offsets.MonthEnd(-1)
    elif delta_type[:3] == "YoY":
        compareDate = selected_date + pd.offsets.MonthEnd(-12)
    elif delta_type[:3] == "YTD":
        compareDate = selected_date + pd.offsets.YearEnd(-1)

    compare_method = delta_type.split(" ")[1].lower()

    metric1 = generate_metric(
        st=col1,
        label="Total Credit Exposure AUD",
        value=df_filtered.query("datadate==@selected_date")["total_exposure_aud"].sum(),
        compare_value=df_filtered.query("datadate==@compareDate")[
            "total_exposure_aud"
        ].sum(),
        compare_method=compare_method,
        delta_suffix=delta_type[:3],
        value_prefix='$'
    )

    metric2 = generate_metric(
        st=col2,
        label="Total On-B/S Exposure AUD",
        value=df_filtered.query("datadate==@selected_date")["on_bs_amount_aud"].sum(),
        compare_value=df_filtered.query("datadate==@compareDate")["on_bs_amount_aud"].sum(),
        compare_method=compare_method,
        delta_suffix=delta_type[:3],
        value_prefix='$',
    )

    metric3 = generate_metric(
        st=col3,
        label="Total Off-B/S Exposure AUD",
        value=df_filtered.query("datadate==@selected_date")["off_bs_exposure_aud"].sum(),
        compare_value=df_filtered.query("datadate==@compareDate")[
            "off_bs_exposure_aud"
        ].sum(),
        compare_method=compare_method,
        delta_suffix=delta_type[:3],
        value_prefix='$',
        help = 'Include both undrawn and trade finance off-blance sheet outstanding.'
    )

    metric4 = generate_metric(
        st=col4,
        label="Undrawn Amount AUD",
        value=df_filtered.query("datadate==@selected_date")["undrawn_amount_aud"].sum(),
        compare_value=df_filtered.query("datadate==@compareDate")[
            "undrawn_amount_aud"
        ].sum(),
        compare_method=compare_method,
        delta_suffix=delta_type[:3],
        value_prefix='$'
    )

    metric5 = generate_metric(
        st=col1,
        label="Number of Client",
        value=df_filtered.query("datadate==@selected_date")["client_name"].nunique(),
        compare_value=df_filtered.query("datadate==@compareDate")["client_name"].nunique(),
        compare_method="absolute",
        delta_suffix=delta_type[:3],
    )

    metric6 = generate_metric(
        st=col2,
        label="Number of Loan Client",
        value=df_filtered.query("datadate==@selected_date & product_category =='Loan'")[
            "client_name"
        ].nunique(),
        compare_value=df_filtered.query(
            "datadate==@compareDate & product_category =='Loan'"
        )["client_name"].nunique(),
        compare_method="absolute",
        delta_suffix=delta_type[:3],
    )

    metric7 = generate_metric(
        st=col3,
        label="Number of Trade Finance Client",
        value=df_filtered.query(
            "datadate==@selected_date & product_category =='Trade Finance'"
        )["client_name"].nunique(),
        compare_value=df_filtered.query(
            "datadate==@compareDate & product_category =='Trade Finance'"
        )["client_name"].nunique(),
        compare_method="absolute",
        delta_suffix=delta_type[:3],
    )

    metric8 = generate_metric(
        st=col4,
        label="Number of Corporate Bond Client",
        value=df_filtered.query(
            "datadate==@selected_date & product_category =='Corporate Bond'"
        )["client_name"].nunique(),
        compare_value=df_filtered.query(
            "datadate==@compareDate & product_category =='Corporate Bond'"
        )["client_name"].nunique(),
        compare_method="absolute",
        delta_suffix=delta_type[:3],
    )

    metric9 = generate_metric(
        st=col1,
        label="Number of Stage 1 Client",
        value=df_filtered.query("datadate==@selected_date & watchlist_stage=='Stage 1'")[
            "client_name"
        ].count(),
        compare_value=df_filtered.query(
            "datadate==@compareDate & watchlist_stage=='Stage 1'"
        )["client_name"].count(),
        compare_method="absolute",
        delta_suffix=delta_type[:3],
    )

    metric10 = generate_metric(
        st=col2,
        label="Number of Stage 2 Client",
        value=df_filtered.query("datadate==@selected_date & not watchlist_stage.isna()")[
            "client_name"
        ].count(),
        compare_value=df_filtered.query(
            "datadate==@compareDate & not watchlist_stage.isna()"
        )["client_name"].count(),
        compare_method="absolute",
        delta_suffix=delta_type[:3],
    )

    metric11 = generate_metric(
        st=col3,
        label="Watchlist Exposure as % of CP",
        value=df_filtered.query("datadate==@selected_date & not watchlist_stage.isna()")[
            "total_exposure_aud"
        ].sum()
        / df_filtered.query("datadate==@selected_date")["total_exposure_aud"].sum(),
        compare_value=df_filtered.query(
            "datadate==@compareDate & not watchlist_stage.isna()"
        )["total_exposure_aud"].sum()
        / df_filtered.query("datadate==@compareDate")["total_exposure_aud"].sum(),
        compare_method="percentage_point",
        delta_suffix=delta_type[:3],
        value_format="percentage",
    )


    metric12 = generate_metric(
        st=col4,
        label="Potential Risk Ratio",
        value=df_filtered.query("datadate==@selected_date & watchlist_stage=='Stage 2'")[
            "total_exposure_aud"
        ].sum()
        / df_filtered.query("datadate==@selected_date")["total_exposure_aud"].sum(),
        compare_value=df_filtered.query(
            "datadate==@compareDate & watchlist_stage=='Stage 2'"
        )["total_exposure_aud"].sum()
        / df_filtered.query("datadate==@compareDate")["total_exposure_aud"].sum(),
        compare_method="percentage_point",
        delta_suffix=delta_type[:3],
        value_format="percentage",
        help ='Watchlist stage 2 exposure as percentage of total credit portfolio'
    )



    metric13 = generate_metric(
        st=col1,
        label="Monthly Total FC AUD",
        value=df_filtered.loc[(df['datadate']==selected_date) & (df['start_date']> df['datadate'] + pd.offsets.MonthEnd(-1))][
            "total_exposure_aud"
        ].sum(),
        compare_value=df_filtered.loc[(df['datadate']==compareDate) & (df['start_date']> df['datadate'] + pd.offsets.MonthEnd(-1))][
            "total_exposure_aud"
        ].sum(),
        compare_method=compare_method,
        delta_suffix=delta_type[:3],
        value_prefix='$',
        help = 'The aggregated value of all deals financially closed in the selected month, regardless of deal types (refinance/new deal) and product category (loan/trade fianance/corporate bond).'

        )

    metric14 = generate_metric(
        st=col2,
        label="Monthly Loan FC AUD",
        value=df_filtered.loc[(df['datadate']==selected_date) & (df['start_date']> df['datadate'] + pd.offsets.MonthEnd(-1)) & (df['product_category']=='Loan')][
            "total_exposure_aud"
        ].sum(),
        compare_value=df_filtered.loc[(df['datadate']==compareDate) & (df['start_date']> df['datadate'] + pd.offsets.MonthEnd(-1)) & (df['product_category']=='Loan')][
            "total_exposure_aud"
        ].sum(),
        compare_method=compare_method,
        delta_suffix=delta_type[:3],
        value_prefix='$',
        help = 'The aggregated value of all loan deals financially closed in the selected month, regardless of deal types (refinance/new deal).'
        )
        
    metric15 = generate_metric(
        st=col3,
        label="Monthly New Loan FC AUD",
        value=df_filtered.loc[(df['datadate']==selected_date) & (df['start_date']> df['datadate'] + pd.offsets.MonthEnd(-1)) & (df['product_category']=='Loan') & (df['loan_purpose']!='Internal Refinance'
        )][
            "total_exposure_aud"
        ].sum(),
        compare_value = df_filtered.loc[(df['datadate']==compareDate) & (df['start_date']> df['datadate'] + pd.offsets.MonthEnd(-1)) & (df['product_category']=='Loan') & (df['loan_purpose']!='Internal Refinance'
        )][
            "total_exposure_aud"
        ].sum(),
        compare_method=compare_method,
        delta_suffix=delta_type[:3],
        value_prefix='$',
        help = 'The aggregated value of all new loan deals financially closed in the selected month, excluding internal refinances.'
        )

    metric16 = generate_metric(
        st=col4,
        label="Loan Utilisation Rate",
        value=df_filtered.query("datadate==@selected_date & product_category=='Loan'")[
            "on_bs_amount_aud"
        ].sum()
        / df_filtered.query("datadate==@selected_date & product_category=='Loan'")["total_exposure_aud"].sum(),
        compare_value=df_filtered.query("datadate==@compareDate & product_category=='Loan'")[
            "on_bs_amount_aud"
        ].sum()
        / df_filtered.query("datadate==@compareDate & product_category=='Loan'")["total_exposure_aud"].sum(),
        compare_method="percentage_point",
        delta_suffix=delta_type[:3],
        value_format="percentage",
        help ='Loan drawn/ (Loan drawn + Loan undrawn)'
    )


def generate_sidebar_filter_form(df):
    with st.sidebar.form("slicer"):
        # filters
        st.subheader("Filters:")
        filt_dataDate = st.selectbox(
            label="Data Date",
            options=df["datadate"]
            .sort_values(ascending=False)
            .dt.strftime("%Y-%m-%d")
            .unique(),
        )
        filt_clientName = st.multiselect(
            "Client Name", options=sorted(df["client_name"].unique())
        )
        filt_productCate = st.multiselect(
            label="Product Category",
            options=sorted(df["product_category"].unique()),
        )
        filt_productType = st.multiselect(
            label="Product Type",
            options=sorted(df["product_type"].unique()),
        )
        filt_anzsicDiv = st.multiselect(
            label="ANZSIC Division", options=sorted(df["anzsic_2006_division"].unique())
        )
        filt_anzsicSubdiv = st.multiselect(
            label="ANZSIC Subdivision",
            options=sorted(df["anzsic_2006_subdivision"].unique()),
        )
        filt_anzsicGroup = st.multiselect(
            label="ANZSIC Industry Group", options=sorted(df["anzsic_2006_group"].unique())
        )
        filt_hoSector = st.multiselect(
            label="HO Industry Classification",
            options=df["ho_industry_classification"].unique(),
        )
        filt_cptSector = st.multiselect(
            label="CPT Industry Classification",
            options=df["cpt_industry_classification"].unique(),
        )
        filt_watchlist = st.multiselect(
            label="Watchlist Stage", options=df["watchlist_stage"].dropna().unique()
        )
        filt_days2maturity = st.slider(
            label="Days to Maturity",
            min_value=0,
            max_value=10000,
            value=(0, 10000),
            step=30,
        )
        filt_termDays = st.slider(
            label="Term Days",
            min_value=0,
            max_value=10000,
            value=(0, 10000),
            step=30,
        )
        filt_ppp = st.checkbox("Show PPP clients only", value=False)
        filt_cre = st.checkbox("Show CRE clients only", value=False)
        filt_rp = st.checkbox("Exclude all trade finance risk particiation", value=False)

        st.markdown("&nbsp;")
        col1, col2 = st.columns([6, 2.5])
        btn_apply = col2.form_submit_button("Apply")   
        # btn_reset = col1.form_submit_button("Reset")

        filter_dict = {
            'filt_dataDate':filt_dataDate,
            'filt_clientName': filt_clientName,
            'filt_productCate': filt_productCate,
            'filt_productType': filt_productType, 
            'filt_anzsicDiv':filt_anzsicDiv,
            'filt_anzsicSubdiv':filt_anzsicSubdiv,
            'filt_anzsicGroup':filt_anzsicGroup,
            'filt_hoSector':filt_hoSector,
            'filt_cptSector':filt_cptSector, 
            'filt_watchlist':filt_watchlist,
            'filt_days2maturity':filt_days2maturity,
            'filt_termDays':filt_termDays,
            'filt_ppp':filt_ppp,
            'filt_cre':filt_cre,
            'filt_rp':filt_rp,
            'btn_apply':btn_apply
            }

        return filter_dict

def generate_trend_chart(df_filtered, selected_date):
    st.markdown("&nbsp;")
    st.subheader("PORTFOLIO TREND", help = "Examine the ongoing progression of the branch's credit portfolio up to a specified month" )

    col1, col2, col3 = st.columns(3, gap="small")
    chart_type = col1.radio(
        "Select chart type", options=["Bar chart", "Area chart"], horizontal=True
    )
    col1, col2, col3 = st.columns(3, gap="small")
    xaxis = col1.selectbox("Select x-axis", options=["Month", "Quarter", "Year"])
    yaxis = col2.selectbox(
        "Select y-axis",
        options=[
            "Total Credit Exposure AUD",
            "On Balance Sheet Exposure AUD",
            "Off Balance Sheet Exposure AUD",
            "Undrawn Amount AUD",
        ],
    )

    list_category = sorted(
            [
                col.replace("_", " ").title()
                for col, dtype in zip(df_filtered.columns, df_filtered.dtypes)
                if dtype == "object"
            ]
    )
    legend = col3.selectbox(
        "Select 3rd dimension (optional)",
        options=list_category,
        index=None,
    )

    agg_dict = {
        "Total Credit Exposure AUD": "total_exposure_aud",
        "On Balance Sheet Exposure AUD": "on_bs_amount_aud",
        "Off Balance Sheet Exposure AUD": "off_bs_exposure_aud",
        "Undrawn Amount AUD": "undrawn_amount_aud",
    }

    agg_col_name = agg_dict[yaxis]

    if xaxis == "Month":
        df_chart = df_filtered.query(
            "(datadate.dt.is_month_end & datadate < @selected_date) or datadate==@selected_date "
        )
    elif xaxis == "Quarter":
        df_chart = df_filtered.query(
            "(datadate.dt.is_quarter_end & datadate < @selected_date) or datadate==@selected_date "
        )
    elif xaxis == "Year":
        df_chart = df_filtered.query(
            "(datadate.dt.is_year_end & datadate < @selected_date) or datadate==@selected_date "
        )

    if legend:
        color_col_name = "_".join(legend.split(" ")).lower()
        df_chart = (
            df_chart.groupby(["datadate", color_col_name])[[agg_col_name]]
            .sum()
            .reset_index()
            .sort_values("datadate")
        )
    else:
        color_col_name = None
        df_chart = (
            df_chart.groupby(["datadate"])[[agg_col_name]]
            .sum()
            .reset_index()
            .sort_values("datadate")
        )

    if chart_type == "Bar chart":
        plot = px.histogram(
            df_chart,
            x="datadate",
            y=agg_col_name,
            color=color_col_name,
            title=f"{yaxis} by {xaxis}",
            text_auto=True,
        )

        plot.update_layout(
            yaxis_title="Total Credit Exposure AUD",
            xaxis=dict(
                tickmode="array",
                tickvals=df_chart["datadate"],
                ticktext=df_chart["datadate"].dt.strftime("%b %Y"),
                tickangle=45,  # Angle of the tick labels
                tickfont=dict(size=12),  # Adjust font size if needed
                title=None,
                type="category",
            ),
        )
        plot.update_traces(texttemplate="%{y:.3s}")


    elif chart_type == "Area chart":
        plot = px.area(
            df_chart,
            x="datadate",
            y=agg_col_name,
            color=color_col_name,
            groupnorm="fraction",
            title=f"{yaxis} % by {xaxis}",
        )
        plot.update_layout(
            yaxis=dict(tickformat="%", title="% of Total Credit Exposure"),
            xaxis_title=None,
            hovermode="x unified",
        )

    st.plotly_chart(plot, use_container_width=True)

def generate_dynamic_chart(df_filtered, selected_date):
    st.markdown("&nbsp;")
    st.subheader("DYNAMIC CHART",
        help='''This dynamic chart provides multifaceted insights into the 
        composition of the branch's credit portfolio at any given month''')
    col1, col2 = st.columns([2, 1], gap="small")
    chart_type = col1.radio(
        "Select chart type",
        options=["Bar chart", "Horizontal bar chart"],
        horizontal=True,
    )

    sort = col2.selectbox(
        "Sort by",
        options=[
            "Total descending",
            "Total ascending",
            "Category descending",
            "Category ascending",
        ],
    )

    col1, col2, col3 = st.columns(3, gap="small")

    list_category = sorted(
        [
            col.replace("_", " ").title()
            for col, dtype in zip(df_filtered.columns, df_filtered.dtypes)
            if dtype == "object"
        ]
    )


    category = col1.selectbox(
        "Category",
        options=list_category,
        key="chart_category",
    )
    agg_col = col2.selectbox(
        "Aggregated value",
        options=[
            "Total Credit Exposure AUD",
            "On Balance Sheet Exposure AUD",
            "Off Balance Sheet Exposure AUD",
            "Undrawn Amount AUD",
        ],
        key="agg_col",
    )
    

    color = col3.selectbox(
            "3rd dimension (optional)",
            options=list_category,
            key="chart_color", 
            index=None,
    )

    max_category = col1.number_input(
        "Max number of categories on chart", value=10, min_value=10
    )


    include_others = st.checkbox(
        "Show the rest as others if the number of categories is greater than the max allowed",
        True,
    )

    agg_dict = {
    "Total Credit Exposure AUD": "total_exposure_aud",
    "On Balance Sheet Exposure AUD": "on_bs_amount_aud",
    "Off Balance Sheet Exposure AUD": "off_bs_exposure_aud",
    "Undrawn Amount AUD": "undrawn_amount_aud",
}
    # convert back to lower case to matach database column name
    agg_col_name = agg_dict[agg_col]
    category_col_name=category.replace(" ", "_").lower()
    

    if color:
        color_col_name = color.replace(" ", "_").lower()
        if category == color: 
            group_by_col_name = [category_col_name]
        else:
            group_by_col_name = [category_col_name, color_col_name]
    else:
        group_by_col_name = [category_col_name]


    last_month_end = pd.to_datetime(selected_date) + pd.offsets.MonthEnd(-1)
    same_date_last_year = pd.to_datetime(selected_date) + pd.offsets.MonthEnd(-12)
    last_year_end = pd.to_datetime(selected_date) + pd.offsets.YearEnd(-1)

    df_curMonth = (
        df_filtered.query("datadate==@selected_date")
        .groupby(group_by_col_name)[[agg_col_name]]
        .sum()
        .reset_index()
    )
    df_MoM = (
        df_filtered.query("datadate==@last_month_end")
        .groupby(group_by_col_name)[[agg_col_name]]
        .sum()
        .reset_index()
    )
    df_YoY = (
        df_filtered.query("datadate==@same_date_last_year")
        .groupby(group_by_col_name)[[agg_col_name]]
        .sum()
        .reset_index()
    )
    df_YTD = (
        df_filtered.query("datadate==@last_year_end")
        .groupby(group_by_col_name)[[agg_col_name]]
        .sum()
        .reset_index()
    )

    # If more than 10 categories, cum sum the rest as others
    df_chart = df_curMonth.sort_values(agg_col_name, ascending=False)

    num_category = len(df_filtered.query("datadate==@selected_date")
        .groupby(group_by_col_name[0]))

    if num_category > max_category:
        # dropna as some column may have nan value, such as watchlist status, CRE category etc
        top_list = df_filtered.query("datadate==@selected_date").dropna(subset=group_by_col_name).groupby(group_by_col_name[0])[agg_col_name].sum().nlargest(max_category).index

        df_chart_top = df_curMonth[df_curMonth[category_col_name].isin(top_list)]
        df_chart_others = df_curMonth[~df_curMonth[category_col_name].isin(top_list)]
        df_chart_others[category_col_name] = 'Others'

        if include_others:
            df_chart = pd.concat([df_chart_top, df_chart_others], ignore_index=True)
        else:
            df_chart = df_chart_top

    if chart_type == "Bar chart":
        fig = px.histogram(
            df_chart,
            x=group_by_col_name[0],
            y=agg_col_name,
            color = group_by_col_name[1] if len(group_by_col_name)>1 else None,
            text_auto=True,
            title=f"{agg_col} by {category}",
        )

        fig.update_layout(
            xaxis={"categoryorder": sort.lower(), "title": None},
            yaxis_title=agg_col,
        )
        fig.update_traces(texttemplate="%{y:.3s}")
    elif chart_type == "Horizontal bar chart":
        fig = px.histogram(
            df_chart,
            x=agg_col_name,
            y=group_by_col_name[0],
            color = group_by_col_name[1] if len(group_by_col_name)>1 else None,
            orientation="h",
            text_auto=True,
            title=f"{agg_col} by {category}",
        )
        fig.update_layout(
            xaxis_title=agg_col,
            yaxis={"categoryorder": sort.lower(), "title": None},
        )
        fig.update_traces(texttemplate="%{x:.3s}")
    elif chart_type == "Pie chart":
        fig = px.pie(
            df_chart,
            names= group_by_col_name[0],
            values=agg_col_name,
            title=f"{agg_col} by {category}",
        )
        # fig.update_traces(textposition='inside', textinfo='percent+label')


    st.plotly_chart(fig)

    st.markdown("***Data view:***")

    df_table = df_curMonth.sort_values(agg_col_name, ascending=False)
    df_table["%_of_total"] = df_table[agg_col_name] / df_table[agg_col_name].sum()
    df_table["%_of_total"] = df_table["%_of_total"].apply(
        lambda x: f"{x * 100:.2f}%" if not pd.isna(x) else None
    )
    df_table["%_of_credit_portfolio"] = df_table[agg_col_name] / (
        df.query("datadate==@selected_date")[agg_col_name].sum()
    )
    df_table["%_of_credit_portfolio"] = df_table["%_of_credit_portfolio"].apply(
        lambda x: f"{x * 100:.2f}%" if not pd.isna(x) else None
    )
    list_dfs = [df_table, df_MoM, df_YoY, df_YTD]

    for i in range(1, len(list_dfs)):
        suffix = [None, "_LME", "_SPLY", "_LYE"][i]
        delta_col_name = [None, "MoM Δ %", "YoY Δ %", "YTD Δ %"][i]
        df_table = pd.merge(
            df_table,
            list_dfs[i],
            on=group_by_col_name,
            how="left",
            suffixes=(None, suffix),
        )
        df_table[delta_col_name] = (
            df_table[agg_col_name] / df_table[agg_col_name + suffix] - 1
        )
        # df =df.style.format({delta_col_name:lambda x: '{:,.1f} %'.format(x) if not pd.isna(x) else None})
        df_table[delta_col_name] = df_table[delta_col_name].apply(
            lambda x: f"{x * 100:.2f}%" if not pd.isna(x) else None
        )

    # df_table[agg_col_name]= df_table[agg_col_name].astype(float)
    df_table.columns = df_table.columns.str.upper()

    # column_config ={agg_col_name.upper():st.column_config.ProgressColumn(width='medium',format='$%.2f',min_value=0,max_value=df_table[agg_col_name.upper()].max())}
    st.dataframe(
        df_table,
        hide_index=True,
        # column_config =column_config
    )
    st.caption(
        "Note:  The suffixes of LME, SPLY and LYE in above table mean last month end, same period last year, last year end respectively"
    )






#----------------------------------Scripts starting from here---------------------------------------------------------


if __name__ == '__main__':
    # set streamli page configuration
    st.set_page_config(page_title="Hello", page_icon="👋", layout="wide")


    # Creating a login widget
    file_path = Path(__file__).parent / 'config.yaml'
    with open(file_path) as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized'])

    authenticator.login('Login','sidebar')

    # Authenticating users
    if st.session_state["authentication_status"] is False:
        st.image("icbc_logo_long.png", width=400)
        st.header('ICBC Sydney Credit Portfolio BI Platform')
        st.divider()
        st.sidebar.error('Username/password is incorrect')
        forgot_password = st.sidebar.toggle('Forgot username/password')
        # Creating a forgot password widget
        if forgot_password:
            try:
                username_of_forgotten_password, email_of_forgotten_password, new_random_password = authenticator.forgot_password('Forgot password')
                if username_of_forgotten_password:
                    st.success('New password to be sent securely')
                    with open(file_path, 'w') as file:
                        yaml.dump(config, file, default_flow_style=False)
                    # Random password should be transferred to user securely
                    msg = f'username: {username_of_forgotten_password}\n\npassword: {new_random_password}'
                    send_email(email_of_forgotten_password,'Reset Password', msg)
                else:
                    st.error('Username not found')

            except Exception as e:
                st.error(e)
        # Creating a forgot username widget
        else:
            try:
                username_of_forgotten_username, email_of_forgotten_username = authenticator.forgot_username('Forgot username')
                if username_of_forgotten_username:
                    st.success('Username to be sent securely')
                    # Username should be transferred to user securely
                    msg = f'username: {username_of_forgotten_username}'
                    send_email(email_of_forgotten_username,'Forgot Username', msg)
                else:
                    st.error('Email not found')
            except Exception as e:
                st.error(e)
                

    elif st.session_state["authentication_status"] is None:
        st.image("icbc_logo_long.png", width=400)
        st.header('ICBC Sydney Credit Portfolio BI Platform')
        st.divider()
        st.sidebar.warning('Please enter your username and password')
        # Creating a new user registration widget
        try:
            if authenticator.register_user('Register user', preauthorization=False):
                st.success('User registered successfully')
                with open(file_path, 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(e) 
        st.caption('*All registrations must be pre-authorised by DGM. After approval, please contact creditportfolio@icbc.com.au.')


    elif st.session_state["authentication_status"]:

        st.image("icbc_logo_long.png", width=400)
        st.header('ICBC Sydney Credit Portfolio BI Platform')
        st.divider()
        st.sidebar.title(f'Welcome *{st.session_state["name"].split()[0]}* ! 👋')
        authenticator.logout('Logout', 'sidebar', key='unique_key')
        with st.sidebar:
            menu = option_menu(
                menu_title = None,
                options=['Dashboard', 'Help','Account Setting',],
                default_index=0,
                menu_icon='house',
                icons=['graph-up-arrow','info-circle','person',],
                styles={
                    "container": {"padding": "0!important", "background-color": "#fafafa"},
                    "menu-title": {"color": "black", "font-size": "15px"},
                    "icon": {"color": "black", "font-size": "20px"}, 
                    "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                    "nav-link-selected": {"font-size": "15px","background-color": "#9C9999",},
                })

        # Creating a password reset widget
        if menu == 'Account Setting':
            account_setting = st.selectbox('Account Setting',options=['Change password', 'Change user details'])
            if account_setting == 'Change password':
                try:
                    if authenticator.reset_password(st.session_state["username"], 'Reset password','main'):
                        st.success('Password changed successfully')
                        with open(file_path, 'w') as file:
                            yaml.dump(config, file, default_flow_style=False)
                except Exception as e:
                    st.error(e)
        # Creating an update user details widget
            elif account_setting == 'Change user details':
                try:
                    if authenticator.update_user_details(st.session_state["username"], 'Update user details','main'):
                        st.success('Entries updated successfully')
                        with open(file_path, 'w') as file:
                            yaml.dump(config, file, default_flow_style=False)
                except Exception as e:
                    st.error(e)
        
        elif menu == 'Help':
        
            st.info('''
                :blue[**First time to this platform?**] 

                ***No worries, this video will cover :rainbow[everything] you need to know.*** :balloon:
                ''')
            video_file = open('example_video.mp4', 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

        elif menu == 'Dashboard':    

            with st.spinner("Retriving data..."):
                # get data
                df = get_data()
                df_filtered = df

            with st.expander("About this app"):
                st.info(
                    '''
                    - Explore ICBC Sydney branch's **Credit Portfolio** effortlessly with our visualization app.
                    - Track and gain valuable insights into branch-specific credit performance, 
                    empowering informed decision-making for optimal financial outcomes.
                    - Leveraging the robust filtering capabilities in the left sidebar to delve into more detailed and nuanced aspects.
                    '''
                )

            # generate side bar filter form, and unpack the data
            filter_dict =generate_sidebar_filter_form(df)

            # data picker
            selected_date = pd.to_datetime(filter_dict['filt_dataDate'])

            # filter data except for date
            if filter_dict['btn_apply']:
                st.balloons()
                st.sidebar.success('Filters applied successfully')
                df_filtered = filter_data_except_date(
                    df,
                    filter_dict['filt_clientName'],
                    filter_dict['filt_productCate'],
                    filter_dict['filt_productType'],
                    filter_dict['filt_anzsicDiv'],
                    filter_dict['filt_anzsicSubdiv'],
                    filter_dict['filt_anzsicGroup'],
                    filter_dict['filt_hoSector'],
                    filter_dict['filt_cptSector'],
                    filter_dict['filt_watchlist'],
                    filter_dict['filt_days2maturity'],
                    filter_dict['filt_termDays'],
                    filter_dict['filt_ppp'],
                    filter_dict['filt_cre'],
                    filter_dict['filt_rp'],
                    )
            
            # generate metrics
            generate_metrics(df_filtered, selected_date)

            # generate trend chart
            generate_trend_chart(df_filtered, selected_date)

            # generate dynamic chart
            generate_dynamic_chart(df_filtered, selected_date)


