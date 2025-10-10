const digitForm = document.getElementById('digitForm');
const imageForm = document.getElementById('imageForm');

digitForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(digitForm);
    const response = await fetch('/predict-digit', {
        method: 'POST',
        body: formData
    });
    const data = await response.json();
    document.getElementById('digitResult').innerText = "Predicted Digit: " + data.prediction;
});

imageForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(imageForm);
    const response = await fetch('/predict-image', {
        method: 'POST',
        body: formData
    });
    const data = await response.json();
    document.getElementById('imageResult').innerText = "Predicted Image: " + data.prediction;
});
