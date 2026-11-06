# Week 3: Advanced Data Analysis and Visualization in Logistics

Yuva Intern data science task. Exploratory data analysis (EDA) and visualization of a hypothetical logistics dataset (6,000 shipments, 3 hubs, 3 carriers, 3 transport modes).

## Files
| File | Description |
|------|-------------|
| `Week3_Data_Analysis_Visualization_Report.docx` | Full report with charts, insights and recommendations |
| `eda_visualization.py` | Python script that builds the dataset, runs EDA and creates the charts |
| `logistics_dataset.csv` | The generated dataset |
| `charts/` | The 7 charts (PNG) used in the report |

## Charts
1. Distribution of delivery time (histogram)
2. Monthly shipment volume and delay rate (bar + line)
3. Transport cost by mode (boxplot)
4. Distance vs transport cost (scatter plot)
5. Correlation heatmap
6. Delay rate by carrier and hub (grouped bar)
7. Average cost per km by hub and mode (heatmap)

## Key findings
- Distance is the main cost driver (correlation 0.86); air is 15% of shipments but about 69% of spending
- Delay rate jumps from about 7% to 26% in October to December
- CityCourier and Hub C are the main sources of delay

## Run
```bash
pip install pandas numpy matplotlib seaborn
python eda_visualization.py
```

Note: data is hypothetical and generated for learning purposes.
