function predictscene() {
    const inputImage = document.getElementById('imageInput');
    const image = inputImage.files[0];
    const model = document.getElementById('Select-model').value;

    if(!image) {
        alert("Please select an image first.");
        return;
    }

    document.getElementById("result-text").innerText = "Loading...";
    document.getElementById("attention-map").style.display = 'none';
    document.getElementById("attention-map").src = "#";

    const previewImage = document.getElementById('preview-image');
    previewImage.src = URL.createObjectURL(image);
    previewImage.style.display = 'block';

    const form= new FormData();
    form.append('file', image);
    form.append('model', model);

    document.getElementById('result-text').innerText = "Loading...";

    fetch('https://Ayush0407-scenerecognition.hf.space/run/predict', {
        method: 'POST',
        body: form
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result-text").innerText = "Predicted Scene: " + (data.class || "Unknown");
        // document.getElementById("attention-map").src = data.attention_map;
        // document.getElementById("attention-map").style.display = 'block';

        if (data.top5) {
        const scoresContainer = document.getElementById("scores");
        scoresContainer.innerHTML = "<b>Top 5 Predictions:</b><br>";
        data.top5.forEach((item, index) => {
            scoresContainer.innerHTML += `${index + 1}. ${item.label} — ${item.score}<br>`;
            });
        }

        if (data.attention_map) {
            if (Object.keys(data.attention_map).length === 0) {
                console.warn("Attention map is empty!");
            }
            
            const layerSelect = document.getElementById("layer-select");
            const headSelect = document.getElementById("head-select");

            const keys = Object.keys(data.attention_map);
            const layerHeadPairs = keys.map(k => {
            const parts = k.split("_");
            return { layer: parseInt(parts[1]), head: parseInt(parts[3]), key: k };
        });

    const uniqueLayers = [...new Set(layerHeadPairs.map(p => p.layer))];
    const uniqueHeads = [...new Set(layerHeadPairs.map(p => p.head))];

    // Populate layer dropdown
    layerSelect.innerHTML = "";
    uniqueLayers.forEach(layer => {
        const option = document.createElement("option");
        option.value = layer;
        option.text = `Layer ${layer}`;
        layerSelect.appendChild(option);
    });

    // Populate head dropdown
    headSelect.innerHTML = "";
    uniqueHeads.forEach(head => {
        const option = document.createElement("option");
        option.value = head;
        option.text = `Head ${head}`;
        headSelect.appendChild(option);
    });

    // Store attention_map in global variable
    window.attentionMaps = data.attention_map;

    // Show first available map
    updateAttentionMap();
}
    })
    .catch(error => {
        document.getElementById("result-text").innerText = "Error: " + error;
    });
}

function updateAttentionMap() {
    const layer = document.getElementById("layer-select").value;
    const head = document.getElementById("head-select").value;
    const key = `layer_${layer}_head_${head}`;
    const map = window.attentionMaps?.[key];

    if (map) {
        const attnImage = document.getElementById("attention-map");
        attnImage.src = map;
        attnImage.style.display = "block";
    }
}

function resetOutput() {
    document.getElementById("result-text").innerText = "Waiting for input...";
    document.getElementById("scores").innerHTML = "";
    document.getElementById("attention-map").src = "#";
    document.getElementById("attention-map").style.display = "none";

    document.getElementById("layer-select").innerHTML = "";
    document.getElementById("head-select").innerHTML = "";

    document.getElementById("preview-image").src = "#";
    document.getElementById("preview-image").style.display = "none";

    window.attentionMaps = {};
}
    document.getElementById('Select-model').addEventListener('change', function() {
    resetOutput();  // Clear previous results
});
// Reset output when the page is loaded

    document.getElementById('imageInput').addEventListener('change', function(event) {
    resetOutput(); 
    const previewImage = document.getElementById('preview-image');
    const file = event.target.files[0];
    
    if (file) {
        previewImage.src = URL.createObjectURL(file);
        previewImage.style.display = 'block';
    } else {
        previewImage.src = '#';
        previewImage.style.display = 'none';
    }
});
