import pandas as pd
import streamlit as st

import plotting
import dataset

st.set_page_config(
    page_title="Process Optimization - ML - RSA",
    page_icon="📈",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/kaymal/rsa-app",
        "Report a bug": "https://github.com/kaymal/rsa-app/issues",
        "About": "Turgut Kaymal https://github.com/kaymal",
    },
)


def show_sidebar(page: str | None = None) -> str:
    """Show sidebar with navigation options."""
    # set default page
    if (page is None) or ("home" in page.lower()):
        index = 0
    elif "data" in page.lower():
        index = 1
    else:
        print("Page doesn't exist!")
        index = 0

    with st.sidebar:
        options = ["📈 Home", "📊 Data"]
        selected_page = st.radio("Navigation", options, index=index)

        return selected_page


def show_page(page: str) -> None:
    """Show page depending on the selection from the sidebar.

    Parameters
    ----------
    page: {"Home", "Data"}
    """
    data = dataset.get_data()
    # get a smaller set of data
    df = dataset.preprocessing(data)

    if page == "📈 Home":
        st.experimental_set_query_params()
        st.markdown(
            "## Feature Optimization Using Response Surface Analysis: A Simple Case Study"
        )
        st.info(
            "💡 The goal of this study is to show how machine failure rate "
            "can be decreased through response surface analysis and optimization of parameters. "
            "The ML pipeline is simplified and most features are ignored for the "
            "purposes of this work."
        )

        st.markdown(
            "The sample dataset below consists of 3 features and 1 target variable. "
            "It is a smaller version of a bigger dataset shown on the 'Data' page. "
            "I assume that using these features, one can predict -with some level of confidence- "
            "whether the result of the process is a success or a failure."
        )
        # show features and target
        tabs = st.tabs(["Data", "Summary Stats"])
        tabs[0].dataframe(df, height=300)
        tabs[1].write(df.describe())

        st.markdown("### Features")
        st.markdown(
            "I divide features/factors into two groups. The first one is "
            "**'controllable features'**. These features can be changed by an "
            "operator/machine (e.g. speed). The second group is **'uncontrollable "
            "(or noise) features'**, which cannot be controlled (e.g. temperature). "
            "In this study, I assume that `rotational speed` and `torque` can be "
            "controlled independently, whereas `temperature` cannot be controlled. "
            "In a real world scenario, this may or may not be the case; however, "
            "for simplicity I assume so."
        )

        cols = st.columns(2)
        cols[0].markdown("**Controllable Features**")
        cols[0].markdown("- rotational_speed_rpm\n- torque_nm")
        cols[1].markdown("**Uncontrollable Features (Noise)**")
        cols[1].write("- temp_ratio")

        # train model
        st.markdown("### Model")
        st.write("I've trained a simple classification model.")
        clf = dataset.train(df)

        st.markdown("### Response Surface")
        st.markdown(
            """
            Ideally, one should desing experiments in order to create a response
            surface and perform analysis on it. However, in an industrial production
            environment it may not be possible. Therefore, we may try to use available 
            sensor/operational data to create a response surface.

            Assuming that temperature factors cannot be controlled, I've created 
            two different levels of "temperature ratio":
            - Between 1.025 and 1.027
            - Between 1.030 and 1.032

            The first chart shows the probability of success depending on the 
            `rotational speed` and `torque`, given that the `temperature ratio` is 
            between 1.025 and 1.027. 

            The second chart, on the other hand, shows the probability of success,
            given that the `temperature ratio` is between 1.030 and 1.032.
            
            One could fit a second order polynomial model and find the **max**
            using an optimization algorithm. Yet again, however, for simplicity
            I've shown the observation that gives the maximum probability of success
            using the model.

            As shown in the plots, given different `temperature ratio` values
            (uncontrollable), we may select different values for
            `rotational speed`/`torque` to achieve a higher probability of success.
            """
        )
        # create dataset for plotting
        # select data by temp
        for temp_min, temp_max in [(1.025, 1.027), (1.030, 1.032)]:
            df_selected = df[(df.temp_ratio > temp_min) & (df.temp_ratio < temp_max)]
            df_plotting = df_selected.join(
                pd.Series(
                    clf.predict_proba(df_selected.iloc[:, :-1])[:, 0],
                    name="p_success",
                    index=df_selected.index,
                )
            )
            # show plot
            plotting.plot_response_3d(
                data=df_plotting, x="rotational_speed_rpm", y="torque_nm", z="p_success"
            )

        st.markdown("### Conclusion")
        st.markdown(
            """
            In an industrial production setup, there may be some parameters we can measure but
            not control (e.g. temperature, moisture), some random factors, and some other 
            parameters that we can control (e.g. mechanical speed of a component, 
            maintenance). I believe that the manufacturing process can be improved 
            by optimizing the controllable factors given all other factors.
            This can be achieved by giving automated suggestions to operators,
            as well selecting the best parameter values automatically within the system.
            """
        )
    elif page == "📊 Data":
        st.experimental_set_query_params()
        st.markdown("## Full Dataset")
        st.markdown(
            """
            The synthetically-generated dataset contains observations of machine operations/sensors
            that will help predicting machine failure. The features include 
            rotational speed [rpm], torque [Nm], air temperature [K], and 
            process temperature [K].
            """
        )
        st.write(data)
        st.markdown(
            """
            * **Type**: consisting of a letter L, M, or H for low, medium and high as product quality variants.
            * **air temperature [K]**: generated using a random walk process later normalized to a standard deviation of 2 K around 300 K.
            * **process temperature [K]**: generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
            * **rotational speed [rpm]**: calculated from a power of 2860 W, overlaid with a normally distributed noise.
            * **torque [Nm]**: torque values are normally distributed around 40 Nm with a Ïƒ = 10 Nm and no negative values. 
            * **tool wear [min]**: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process.
            * **machine failure**: whether the machine has failed in this particular datapoint for any of the following failure modes are true.
            """
        )
        st.markdown("## Licence")
        st.write(
            "[Database Contents License (DbCL) v1.0]"
            "(http://opendatacommons.org/licenses/dbcl/1.0/)"
        )
        st.markdown("## Reference")
        st.markdown(
            "- [Machine Failure Predictions]"
            "(https://www.kaggle.com/datasets/dineshmanikanta/machine-failure-predictions)\n"
            "- [Binary Classification of Machine Failures]"
            "(https://www.kaggle.com/competitions/playground-series-s3e17/data)"
        )


if __name__ == "__main__":
    # get page from query params
    query_params = st.experimental_get_query_params()
    if "page" in query_params:
        page = query_params.get("page")[0]
    else:
        page = None

    # show sidebar with page options
    selected_page = show_sidebar(page=page)

    # show selected page on the main container
    show_page(selected_page)
