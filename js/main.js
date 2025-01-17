var text = document.getElementById("text")

var finishButton = document.getElementById("finishButton")

var clicks = []

map.on('click', function (e) {
    if (clicking){
        let click = e.latlng
        clicks.push(click)
        L.marker([click.lat, click.lng]).addTo(map);
        finishButton.style.display = "flex"

    }
});

function finishClicks(){
    if (mode == "path"){

        text.innerHTML = "Processing..."
        waitForDOMUpdate(0).then(r => {

            for (let i = 0; i < clicks.length - 1; i++){
                let startNode = getNearestNode(clicks[i].lat, clicks[i].lng, graph);
                let goalNode = getNearestNode(clicks[i + 1].lat, clicks[i + 1].lng, graph);
                console.log(startNode, goalNode)
                path = aStar(startNode, goalNode, graph)
                console.log(path)
                coordinates = [...coordinates, ...path]
            }
            // Convert the path into a Leaflet polyline and add it to the map
            text.innerHTML = "Press add button to start"
            finish()
        })

    }else{
        bbox = clicks.map(c => [c.lat, c.lng])
        if (hasIntersectingEdges(bbox)){
            bbox = orderPointsMinimizingAngles(bbox)
        }
        bbox.push(bbox[0])
        drawBoundingBox(map, bbox)
        waitForDOMUpdate(1000).then(r => {
            coordinates = solveCPP(bbox)
            finish()
        })
    }
}

function finish(){
    var polyline = convertToPolyline(coordinates)
    var bounds = polyline.getBounds();
    map.fitBounds(bounds)
    finishButton.style.display = "none"
}

var polylineGroup = L.featureGroup().addTo(map);

var penalty = 2

var route = []
var coordinates = []