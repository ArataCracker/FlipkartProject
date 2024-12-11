const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureInput = document.getElementById('captured_image');

navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;
    })
    .catch((err) => {
        alert('Could not access camera: ' + err);
    });

document.querySelector('button[type="submit"]').addEventListener('click', (e) => {
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    captureInput.value = canvas.toDataURL('image/jpeg');
});
