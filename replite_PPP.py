import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Read in the CSV files
# -------------------------------

# Read PPP_Base and make a copy
PPP = pd.read_csv("PPP_Base.csv")

#PPP["Date"] = pd.to_datetime(PPP["Date"])

NEW = 2

if NEW==1:
    data = pd.read_csv("Data_Base_init.csv")  

if NEW==2:
    data = pd.read_csv("Data_Base.csv")

# -------------------------------
# 2. Create and modify data_N   
# -------------------------------

# Create a 12x4 DataFrame filled with ones and name columns accordingly
data_N = pd.DataFrame(np.ones((12, 4)), columns=["tcm1yN", "EGN", "cciN", "EG"])

# In Julia, the loop sets all rows of the "EG" column to 0.133.
data_N["EG"] = 0.133

# For each row, recalc the values using data from 'data'.
# (Assumes that data has columns "tcm1y" and "cci".)
for i in range(12):
    # Compute standardized tcm1y value
    data_N.at[i, "tcm1yN"] = (data.loc[i, "tcm1y"] - 3.28) / 2.50
    # Compute standardized EGN using the already-set EG value
    data_N.at[i, "EGN"] = (data_N.at[i, "EG"] - 0.072) / 0.196
    # Compute standardized cciN value
    data_N.at[i, "cciN"] = (data.loc[i, "cci"] - 97.0) / 25.3

# (In Julia, vscodedisplay shows the DataFrame; in Python we print it.)
# print("data_N:")
# print(data_N)

# -------------------------------
# 3. Create PPredD and load coefPTdf2182
# -------------------------------

# Create a 12 x 7 numpy array filled with 1.01
PPredD = np.full((12, 7), 1.01)

NLOOP = 12

# Load coefficients from CSV.
# If your CSV file does not have a header row, use header=None.
coefPTdf2182 = pd.read_csv("coefPTdf2182.csv", header=None)
# print("\ncoefPTdf2182:")
# print(coefPTdf2182)
# value = coefPTdf2182.iloc[k, 2]
# print("Value:", value, "Type:", type(value))
# print("4.17 Type:", type(4.17))
# -------------------------------
# 4. Calculate PPredD
# -------------------------------
#display(coefPTdf2182)
# In Julia the indices are one-indexed. Here we use zero-indexing.
# Also note that Julia’s indexing for coefPTdf2182[k,1] corresponds to iloc[k,0] in Python.
# Similarly, data_input_N is assumed to be data_N.
for k in range(NLOOP):
    # Set column 1 (index 0) to constant 4.17
    PPredD[k, 0] = 4.17

    # For clarity, note the following mapping from Julia to Python indices:
    # Julia column 1 -> Python iloc index 0
    # Julia column 3 -> Python iloc index 2
    # Julia column 5 -> Python iloc index 4
    # Julia column 7 -> Python iloc index 6
    # Julia column 9 -> Python iloc index 8, etc.
    #
    # PPredD[k, 3] in Julia means 3rd column → Python index 2.
    PPredD[k, 2] = (coefPTdf2182.iloc[k, 0] +
                    coefPTdf2182.iloc[k, 2] * 4.17 +
                    coefPTdf2182.iloc[k, 4] * data_N.at[k, "tcm1yN"] +
                    coefPTdf2182.iloc[k, 6] * 0.38)

    # PPredD[k, 4] in Julia → Python index 3.
    PPredD[k, 3] = (coefPTdf2182.iloc[k, 0] +
                    coefPTdf2182.iloc[k, 2] * 4.17 +
                    coefPTdf2182.iloc[k, 8] * data_N.at[k, "EGN"] +
                    coefPTdf2182.iloc[k, 10] * 0.26)

    # PPredD[k, 5] in Julia → Python index 4.
    PPredD[k, 4] = (coefPTdf2182.iloc[k, 0] +
                    coefPTdf2182.iloc[k, 2] * 4.17 +
                    coefPTdf2182.iloc[k, 12] * data_N.at[k, "cciN"] +
                    coefPTdf2182.iloc[k, 14] * 0.30)

    # PPredD[k, 6] in Julia → Python index 5.
    PPredD[k, 5] = (coefPTdf2182.iloc[k, 0] +
                    coefPTdf2182.iloc[k, 2] * 4.17)

    # PPredD[k, 7] in Julia → Python index 6.
    PPredD[k, 6] = (coefPTdf2182.iloc[k, 0] +
                    coefPTdf2182.iloc[k, 2] * 4.17 +
                    coefPTdf2182.iloc[k, 4] * data_N.at[k, "tcm1yN"] +
                    coefPTdf2182.iloc[k, 6] * 0.38 +
                    coefPTdf2182.iloc[k, 8] * data_N.at[k, "EGN"] +
                    coefPTdf2182.iloc[k, 10] * 0.26 +
                    coefPTdf2182.iloc[k, 12] * data_N.at[k, "cciN"] +
                    coefPTdf2182.iloc[k, 14] * 0.30)


# print("\nPPredD:")
# print(PPredD)

# print("\nPPP (before PPP calculations):")
# print(PPP)

# -------------------------------
# 5. Calculate PPP (updating PPP DataFrame)
# -------------------------------

# The Julia code updates specific rows and columns of PPP.
# In Julia: PPP[k+7, 7] (7th column, one-indexed) corresponds to PPP.iloc[k+7, 6] (zero-indexed) in Python.
# Similarly, PPredD[k,3] in Julia (3rd column) is PPredD[k,2] in Python, etc.
#
# We also assume that:
#   - data has a column "earn"
#   - PPP has at least 11 columns (with positions matching the Julia code)
for k in range(NLOOP):
    PPP.iloc[k+7, 6]  = 100 * data.loc[k, "earn"] / PPredD[k, 2]  # tcm1yN using forecasted PPredD column 3
    PPP.iloc[k+7, 7]  = 100 * data.loc[k, "earn"] / PPredD[k, 3]  # EG
    PPP.iloc[k+7, 8]  = 100 * data.loc[k, "earn"] / PPredD[k, 4]  # cciN
    PPP.iloc[k+7, 9]  = 100 * data.loc[k, "earn"] / PPredD[k, 5]  # Base (10th column in Julia)
    PPP.iloc[k+7, 10] = 100 * data.loc[k, "earn"] / PPredD[k, 6]  # Full (11th column in Julia)
    PPP.iloc[k+7, 2]  = 100 * data.loc[k, "earn"] / PPredD[k, 0]  # Cetop (3rd column in Julia)

# print("\nPPP (after PPP calculations):")
# print(PPP)

# -------------------------------
# 6. Plotting
# -------------------------------

print(PPP["Date"].dtype)
print(PPP["Date"].head())

import plotly.graph_objects as go

fig = go.Figure()

# Plot other series
fig.add_trace(go.Scatter(x=PPP["Date"],y=PPP["Cetop"], mode='markers+lines',name='Cetop',marker=dict(color='black', size=8), line=dict(color='black', width=3)))
fig.add_trace(go.Scatter(x=PPP["Date"], y=PPP["Base"], mode='lines', name='Base', line=dict(color='black',width=3)))

fig.add_trace(go.Scatter(x=PPP["Date"], y=PPP["tcm1y"], mode='lines', name='tcm1y', line=dict(color='red')))
fig.add_trace(go.Scatter(x=PPP["Date"], y=PPP["EG"], mode='lines', name='EG', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=PPP["Date"], y=PPP["cci"], mode='lines', name='cci', line=dict(color='green')))


# Plot Full: scatter then line (using a blue tone)
fig.add_trace(go.Scatter(x=PPP["Date"], y=PPP["Full"], mode='markers+lines', name='Full', marker=dict(color='blue',size=8), line=dict(color='blue', width=3)))

fig.add_trace(go.Scatter(x=PPP["Date"], y=PPP["Full_init"]+30.0, mode='lines', name='Full_init', line=dict(color='purple', dash='dash', width=3)))

# Plot StdErrP and StdErrM with dashed lines
if NEW==1:
    fig.add_trace(go.Scatter(x=PPP["Date"], y=PPP["StdErrP"], mode='lines', name='StdErrP', line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(x=PPP["Date"], y=PPP["StdErrM"], mode='lines', name='StdErrM', line=dict(color='blue', dash='dash')))



# index_value = 0.01  # or whatever value you want

# for i in range(1, 7):
#     PPP.at[i-1, "MeanP"] = PPP.at[i-1, "Full"] 

# for i in range(7, 20):
#     PPP.at[i-1, "MeanP"] = PPP.at[i-1, "Full"] * (1 + (i-6) * (index_value))

# fig.add_trace(go.Scatter(x=PPP["Date"], y=PPP["MeanP"], mode='lines', name='StdErrP', line=dict(color='blue', dash='dash')))





# Plot actual values again in darkgreen
fig.add_trace(go.Scatter(x=PPP["Date"], y=PPP["actual"], mode='markers+lines', name='actual', marker=dict(color='darkgreen'), line=dict(color='darkgreen', width=3)))

# Optionally add horizontal lines (if needed)
fig.add_hline(y=PPP["actual"].iloc[6], line=dict(color='black', width=3))

date_vline = '2024-12-01'
fig.add_vline(x=date_vline, line_width=3,  line_color="black")


fig.update_layout(
    yaxis_title="sp500",
    title="3MR Reactive Price Prediction App",
    legend=dict(x=0, y=1),
    width=900,
    height=650
)

fig.show()
