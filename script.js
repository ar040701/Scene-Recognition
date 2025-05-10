function predictscene() {
    const inputImage = document.getElementById('imageInput');
    const image = inputImage.files[0];

    if(!image) {
        alert("Please select an image first.");
        return;
    }

    const previewImage = document.getElementById('preview-image');
    previewImage.src = URL.createObjectURL(image);
    previewImage.style.display = 'block';

    const form= new FormData();
    form.append('file', image);

    document.getElementById('result-text').innerText = "Loading...";

    fetch('http://127.0.0.1:8000/predict/', {
        method: 'POST',
        body: form
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result-text").innerText = "Predicted Scene: " + (data.class || "Unknown");
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
