var uploaded = false

const slider = document.getElementById('slider');
const sliderValue = document.getElementById('sliderValue');

slider.addEventListener('mouseup', async function () {
    
    sliderValue.textContent = slider.value / 10;
    penalty = sliderValue.textContent
    
    var path = aStar(route[0], route[1], graph)
    
    // Convert the path into a Leaflet polyline and add it to the map
    path = path == null ? [] : path
    const pathCoordinates = path.map(node => {
        const [lat, lng] = node.split(',').map(Number);
        return [lat, lng];
    });

    var polyline = L.polyline(pathCoordinates, { color: 'green', weight: 4 }).addTo(polylineGroup);
    var bounds = polyline.getBounds();
    map.fitBounds(bounds)
    
});

const button1 = document.getElementById('button1');
const button2 = document.getElementById('button2');

var mode = "path"

function toggleButtons(selectedButton, otherButton) {
    // Highlight the selected button and disable the other
    selectedButton.classList.add('active');
    otherButton.classList.remove('active');

    selectedButton.disabled = true
    otherButton.disabled = false

    selectedButton.style.cursor = "auto"
    otherButton.style.cursor = "pointer"

    selectedButton.style.color = "black"
    otherButton.style.color = "black"

    clearMap()
}

toggleButtons(button1, button2)


// Add event listeners to buttons
button1.addEventListener('click', () => {toggleButtons(button1, button2);mode = "path";});
button2.addEventListener('click', () => {toggleButtons(button2, button1);mode = "fill";});
button1.classList.add('active');


// Optional: Add active styling for the buttons
document.querySelectorAll('.toggleButton').forEach(button => {
    button.addEventListener('click', function () {
        document.querySelectorAll('.toggleButton').forEach(btn => btn.classList.remove('selected'));
        this.classList.add('selected');
    });
});

const button = document.getElementById("downloadButton")
button.addEventListener('click', function () {
    coordinatesToGPX(coordinates);
})

function clearMap(fullClear=false){
    coordinates = []
    map.eachLayer(function (layer) {
        if (layer instanceof L.Marker || layer instanceof L.Polyline || layer instanceof L.Rectangle) {
            if (fullClear || layer.options.id !== 'bounds') {
                map.removeLayer(layer);
            }
        }
    });
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file && file.name.endsWith('.kmz')) {
        const fileInput = document.getElementById("kmzUploader");
        const loadingCircle = document.getElementById("loadingContainer");

        fileInput.disabled = true;
        loadingCircle.style.display = "flex";

        waitForDOMUpdate().then(() => {
            loadKMZFromFile(file);
        });
    } else {
        alert('Please upload a valid KMZ file.');
    }
}

function finishUpload(){
    const loadingCircle = document.getElementById("loadingContainer")
    const removeButton = document.getElementById("removeFileButton")
    const mapElement = document.getElementById("map")

    
    loadingCircle.style.display = "none"
    removeButton.style.display = "flex"
    mapElement.style.cursor = "pointer"

    uploaded = true

    bounds = getBoundsFromGraph(graph)
    bounds = [[bounds[0],bounds[1]],[bounds[2],bounds[3]]]
    console.log(map)
    map.fitBounds(bounds)


    var rectangle = L.rectangle(bounds, {
        color: "black", // Border color
        weight: 2,     // Border width
        fillOpacity: 0 // Transparency for the fill
    })
    rectangle.options.id = 'bounds'
    rectangle.addTo(map)
}

function removeFile() {
    const fileInput = document.getElementById("kmzUploader");
    const removeButton = document.getElementById("removeFileButton");
    const map = document.getElementById("map")

    fileInput.value = ""; // Clear the file input value
    fileInput.disabled = false
    removeButton.style.display = 'none'; // Hide the "Remove File" button
    map.style.cursor = "pointer"

    uploaded = false
    clearMap(true)
}