const form = document.getElementById('uploadForm');
        form.addEventListener('submit', function() {
            const progress = document.querySelector('.progress');
            const progressBar = document.getElementById('progressBar');
            const progressOverlay = document.getElementById('progressOverlay');

            // Show progress overlay and bar
            progress.style.display = 'block';
            progressOverlay.style.display = 'block';

            let width = 0;
            let interval = setInterval(function() {
                console.log(width);
                
                if (width >= 100) {
                    clearInterval(interval);
                } else {
                    width += 10;
                    progressBar.style.width = width + '%';
                    progressBar.innerHTML = width + '%';
                }
            }, 300);
        });

        const fileInput = document.getElementById('fileInput');
        const imagePreviewContainer = document.getElementById('imagePreviewContainer');
        const imagePreview = document.getElementById('imagePreview');
        const resultContainer = document.getElementById("result");

        fileInput.addEventListener('change', function() {
            const file = fileInput.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    imagePreview.src = e.target.result;
                    resultContainer.style.display = 'none';
                    imagePreviewContainer.style.display = 'block';

                };
                reader.readAsDataURL(file);
            }
        });

document.querySelectorAll('.toggle-sidebar').forEach(button => {
  button.addEventListener('click', () => {
    document.getElementById('sidebar').classList.toggle('collapsed');
    document.getElementById('mainContent').classList.toggle('expanded');
  });
});
;