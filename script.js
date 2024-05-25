document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('predictionForm').addEventListener('submit', function(e) {
        e.preventDefault();

       // Get the year and month values from the form
       const year = document.getElementById('year').value;

       // Create the request payload
       const data = JSON.stringify({
         year: year
       });

        makePrediction(data);
    });


    document.getElementById('locationPredictionForm').addEventListener('submit', function(e) {
        e.preventDefault();
        const location = document.getElementById('location').value;
        const year = document.getElementById('locYear').value;
        const data = JSON.stringify({ location: location, year: year });
        makeLocationPrediction(data);
    });
});

function makePrediction(inputData) {
    var apiEndpoint = 'https://0lzjv4f2x6.execute-api.us-east-2.amazonaws.com/prod12/predict12';
   

    fetch(apiEndpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            //'x-api-key': apiKey // Include the API key header if necessary
        },
        body: inputData
    })
    .then(response => response.json())
    .then(data => {
        const body = JSON.parse(data.body);
        const months = body.map(entry => entry.month);
        const temperatures = body.map(entry => parseFloat(entry.predicted_temperature));
        plotData(months, temperatures);
    })
    .catch(error => {
        console.error('Error making prediction:', error);
    });
}


function makeLocationPrediction(inputData) {
    var apiEndpoint = 'https://aaplnfkzga.execute-api.us-east-2.amazonaws.com/Predict/predict';
    fetch(apiEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: inputData
    })
    .then(response => response.json())
    .then(data => {
        const tempPrediction = data.predicted_temperature[0]; // Correctly extract the prediction
        document.getElementById('locationPredictionResult').textContent = 'Prediction: ' + tempPrediction.toFixed(2) + ' °C';
    })
    .catch(error => console.error('Error fetching location prediction:', error));
}


function plotData(months, temperatures) {
    const ctx = document.getElementById('temperatureChart').getContext('2d');
    if (window.myChart) {
        window.myChart.destroy(); // destroy previous chart instance if exists
    }
    window.myChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: months.map(month => `Month ${month}`),
            datasets: [{
                label: 'Predicted Temperature (°C)',
                data: temperatures,
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}
