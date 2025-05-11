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

    fetch('http://127.0.0.1:8000/predict/', {
        method: 'POST',
        body: form
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result-text").innerText = "Predicted Scene: " + (data.class || "Unknown");
        document.getElementById("attention-map").src = data.attention_map;
        document.getElementById("attention-map").style.display = 'block';

        if (data.top5) {
        const scoresContainer = document.getElementById("scores");
        scoresContainer.innerHTML = "<b>Top 5 Predictions:</b><br>";
        data.top5.forEach((item, index) => {
            scoresContainer.innerHTML += `${index + 1}. ${item.label} — ${item.score}<br>`;
            });
        }
        // if (data.attention_map_url) {
        //     const attnImage = document.getElementById("attention-map");
        //     attnImage.src = data.attention_map_url;
        //     attnImage.style.display = "block";
        // }
    })
    .catch(error => {
        document.getElementById("result-text").innerText = "Error: " + error;
    });
}

    document.getElementById('imageInput').addEventListener('change', function(event) {
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
