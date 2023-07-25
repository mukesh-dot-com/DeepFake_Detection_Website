const videoUpload = document.getElementById('videoUpload');
const videoPlayer = document.getElementById('videoPlayer');

videoUpload.addEventListener('change', function(event) {
const file = event.target.files[0];
const reader = new FileReader();

reader.onload = function(event) {
    const videoData = event.target.result;
    videoPlayer.src = videoData;
};

reader.readAsDataURL(file);
});
