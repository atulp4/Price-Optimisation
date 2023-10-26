# Price-Optimisation

### Description
This code appears to be related to a market analysis project. It is written in Python and performs various tasks such as data preprocessing, optimization, and calculation of revenue, gross margin (GM), and other market-related metrics. The code is designed to work with a configuration file (config_skim1.yml) to read input data and set various parameters. A price optimisation and reccommandation engine was developed that successfully integrated customer behavior and csutomer sales data to a linear algorithm technique called COBYLA, to consistently generating price reccomandation for menu items for different global locations. We take in item level data which is data pertaining to each menu item of the QSR chain depending on the location we are doing the optimisation for. Different constraints are provided by the business that have to always be adhered while making price reccomanadations. 


### Dependencies
The code relies on several external Python libraries and modules. You will need to have the following dependencies installed to run the code successfully:
- os
- re
- math
- pandas
- numpy
- openpyxl
- warnings
- ast
- collections
- conjoint_simulator
- conjoint_optimiser

### Key Components
Here are the key components and functionalities of the code:

1. **Configuration Loading**: The code begins by loading configuration settings from the "config_skim1.yml" file.

2. **Logging Setup**: It sets up a logging mechanism to record the execution process, including creating log files.

3. **Data Preparation**: It prepares and processes input data, including item information, respondent data, and model data. This includes data cleaning and formatting.

4. **Estimate Multiplier Calculation**: This section is used for calculating estimate multipliers for GC (Generalized Cost). It involves several data manipulations and calculations.

5. **Optimization**: If a specific configuration setting is enabled, the code proceeds to perform optimization. It considers various constraints and scenarios to optimize the market analysis.

6. **Custom Item Ordering**: There is an option for custom item ordering, where items are ordered based on specific criteria, such as revenue, elasticity, or a custom order list.

7. **Cross Effect Calculation**: The code calculates cross-effects between items, which are pairs of items that influence each other.

8. **Objective Function**: An objective function is defined for the optimization process. It computes various market metrics, such as revenue, gross margin, and GC.

9. **Summary Output**: The code reads input data from an Excel file, applies the objective function, and calculates metrics such as units, revenue, GM, and GC. The results are displayed in the console.

### Running the Code
The code is designed to be run with a specific set of input data and configuration settings. To execute the code successfully, ensure that you have the required input data and configuration file in place.

Please note that this README provides a high-level overview of the code, and a deeper understanding of its functionalities may require knowledge of the specific market analysis project it was developed for.

For questions or additional information, you can contact the author 

### Note
The code appears to be part of a larger project, and the explanation provided here is based on the information available in the code itself. Additional context or documentation may be necessary for a more comprehensive understanding.

    Parameters
    ----------
    df_item : pd.DataFrame
        should have: item_id, price_base, fp_cost, units_base, prices('p_1','p_2','p_3','p_4','p_5'..), weight_item, item_calib_factor
        may have: item_cat, item_channel, item_name, old, is_combo
        if weight_item is blank, a default of 1 is assumed
        if item_calib_factor is blank, a default of 1 is assumed
    df_resp : pd.DataFrame
        should have: resp_id, basket_size, weight_resp, gc_coeff(for KSNR)
        if weight_resp is blank, a default of 1 is assumed
    df_model : pd.DataFrame
        should have:
        resp_id, item_id, model_category, estimate_value, estimate_type (ASC, partworth-<id>,log-linear),     
    cross_effect_item_id,wt,model_calib_factor
        may have: pw_item_id
        if model_category is blank, a items are assumed to be the same model category
        if wt is blank, a default of 1 is assumed
    df_none : pd.DataFrame
        resp_id, item_id, model_category, estimate_value, estimate_type (log-linear), cross_effect_item_id
    scale : str, default="log-log"
        cross effect component sclaing method(log-log , norm, linear)

A config file was created to successfully run it for multiple different cases 

        - config_skim.yml

An integrated script to generate price reccomandations:

        - Price_Optimisation.py 
