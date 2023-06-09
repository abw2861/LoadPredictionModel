### Load Predicton Application Description
This is my capstone project created with Streamlit. The application uses machine learning to predict energy load usage for the Texas Coastal grid area. 
I used an MLP Regressor Model fitted on weather and time features. The application allows the user to enter weather factors along with a specific date and time to receive a load prediction.
The MAPE for this model was 8.113%.


### Datasets
The raw datasets for this project are publically available at:

ERCOT Hourly Load Data: https://www.ercot.com/gridinfo/load/load_hist  
Weather Data: https://mesonet.agron.iastate.edu/request/download.phtml?network=TX_ASOS

### Run the application
At this time, the application requires a local host. To run: 

1. Ensure that Python 3.9+ is installed on your computer.
2. Download LoadPrediction.zip. 
3. Open the file explorer on your computer and navigate to the zip file. 
4. Right-click on the LoadPrediction.zip file, select ‘Extract all..’, select the desired folder where you 
would like to extract the files, and select ‘Extract’. 
5. Navigate to the folder that you selected for file extraction. 
6. Double-click the folder named ‘LoadPrediction’ to open the folder.
7. In the File Explorer’s navigation bar, type ‘cmd’ and press enter. The Windows command 
prompt will open.
8. In the command prompt, type: pip install -r requirements.txt and press Enter. This step only 
needs to be performed once during the initial setup.
9. After the necessary libraries are installed, type into the command prompt: streamlit run 
Home.py and press Enter.
10. If a Windows firewall prompt appears on the screen, click ‘Allow access.’
11. The application should open in your default browser. 

#### Tips
Navigate through the sidebar to view data visualizations on the ‘Analysis’ page.
Navigate through the sidebar to enter weather values and receive a load prediction on the 
‘Predictions’ page. Enter the values into the labelled fields, choose a date and a time, and select 
‘Get Prediction.’ To receive another prediction, simply type and select new values, and select 
the ‘Get Prediction’ button again.
