document.addEventListener("DOMContentLoaded", function () {

    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const dropZoneContent = document.getElementById("drop-zone-content");
    const previewContainer = document.getElementById("preview-container");
    const previewImage = document.getElementById("preview-image");
    const previewFilename = document.getElementById("preview-filename");
    const btnRecognize = document.getElementById("btn-recognize");
    const btnWebcam = document.getElementById("btn-webcam");
    const uploadSection = document.getElementById("upload-section");
    const loadingSection = document.getElementById("loading-section");
    const resultSection = document.getElementById("result-section");
    const resultFound = document.getElementById("result-found");
    const resultNotFound = document.getElementById("result-not-found");
    const resultAdded = document.getElementById("result-added");
    const webcamModal = document.getElementById("webcam-modal");
    const webcamVideo = document.getElementById("webcam-video");
    const webcamCanvas = document.getElementById("webcam-canvas");
    const btnWebcamClose = document.getElementById("btn-webcam-close");
    const btnWebcamCapture = document.getElementById("btn-webcam-capture");
    const addIdentityForm = document.getElementById("add-identity-form");
    const btnReset = document.getElementById("btn-reset");

    let currentFile = null;
    let currentBase64 = null;
    let webcamStream = null;

    dropZone.addEventListener("click", function () {
        fileInput.click();
    });

    dropZone.addEventListener("dragover", function (e) {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", function () {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", function (e) {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith("image/")) return;
        currentFile = file;
        currentBase64 = null;
        const reader = new FileReader();
        reader.onload = function (e) {
            previewImage.src = e.target.result;
            previewFilename.textContent = file.name;
            dropZoneContent.classList.add("hidden");
            previewContainer.classList.remove("hidden");
            btnRecognize.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function setBase64Image(dataUrl) {
        currentFile = null;
        currentBase64 = dataUrl;
        previewImage.src = dataUrl;
        previewFilename.textContent = "Webcam capture";
        dropZoneContent.classList.add("hidden");
        previewContainer.classList.remove("hidden");
        btnRecognize.disabled = false;
    }

    btnWebcam.addEventListener("click", async function () {
        webcamModal.classList.remove("hidden");
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: 640, height: 480 } });
            webcamVideo.srcObject = webcamStream;
        } catch (err) {
            closeWebcam();
        }
    });

    btnWebcamClose.addEventListener("click", closeWebcam);

    webcamModal.addEventListener("click", function (e) {
        if (e.target === webcamModal) closeWebcam();
    });

    function closeWebcam() {
        webcamModal.classList.add("hidden");
        if (webcamStream) {
            webcamStream.getTracks().forEach(function (t) { t.stop(); });
            webcamStream = null;
        }
    }

    btnWebcamCapture.addEventListener("click", function () {
        webcamCanvas.width = webcamVideo.videoWidth;
        webcamCanvas.height = webcamVideo.videoHeight;
        const ctx = webcamCanvas.getContext("2d");
        ctx.translate(webcamCanvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(webcamVideo, 0, 0);
        const dataUrl = webcamCanvas.toDataURL("image/jpeg", 0.92);
        setBase64Image(dataUrl);
        closeWebcam();
    });

    btnRecognize.addEventListener("click", async function () {
        uploadSection.style.display = "none";
        resultSection.classList.add("hidden");
        loadingSection.classList.remove("hidden");

        try {
            let response;
            if (currentFile) {
                const formData = new FormData();
                formData.append("image", currentFile);
                response = await fetch("/api/predict/", { method: "POST", body: formData });
            } else if (currentBase64) {
                response = await fetch("/api/predict/", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ image_base64: currentBase64 }),
                });
            }

            const data = await response.json();
            loadingSection.classList.add("hidden");
            showResult(data);
        } catch (err) {
            loadingSection.classList.add("hidden");
            uploadSection.style.display = "";
        }
    });

    function showResult(data) {
        resultSection.classList.remove("hidden");
        resultFound.classList.add("hidden");
        resultNotFound.classList.add("hidden");
        resultAdded.classList.add("hidden");

        if (data.identity_found && data.identity) {
            resultFound.classList.remove("hidden");
            document.getElementById("result-avatar").src = data.identity.avatar_url || "";
            document.getElementById("result-title").textContent = data.identity.title;
            document.getElementById("result-class").querySelector("span").textContent = data.identity.class_id;
            document.getElementById("result-confidence-text").textContent = data.prediction.confidence + "%";
            setTimeout(function () {
                document.getElementById("result-confidence-bar").style.width = data.prediction.confidence + "%";
            }, 100);
        } else {
            resultNotFound.classList.remove("hidden");
            document.getElementById("notfound-class").textContent = data.prediction.class_id;
            document.getElementById("notfound-confidence").textContent = data.prediction.confidence;
            document.getElementById("new-class-id").textContent = data.prediction.class_id;

            if (currentBase64) {
                document.getElementById("new-avatar-preview").src = currentBase64;
            } else if (currentFile) {
                const reader = new FileReader();
                reader.onload = function (e) {
                    document.getElementById("new-avatar-preview").src = e.target.result;
                };
                reader.readAsDataURL(currentFile);
            }
        }

        resultSection.classList.add("slide-up");
        uploadSection.style.display = "";
        lucide.createIcons();
    }

    addIdentityForm.addEventListener("submit", async function (e) {
        e.preventDefault();
        const title = document.getElementById("new-title").value.trim();
        const classId = document.getElementById("new-class-id").textContent.trim();

        if (!title) return;

        const formData = new FormData();
        formData.append("title", title);
        formData.append("class_id", classId);

        if (currentFile) {
            formData.append("avatar", currentFile);
        } else if (currentBase64) {
            formData.append("avatar_base64", currentBase64);
        }

        try {
            const response = await fetch("/api/add-identity/", { method: "POST", body: formData });
            const data = await response.json();

            if (data.success) {
                resultNotFound.classList.add("hidden");
                resultAdded.classList.remove("hidden");
                document.getElementById("added-message").textContent =
                    '"' + data.identity.title + '" has been added to the database.';
                lucide.createIcons();
            }
        } catch (err) {}
    });

    btnReset.addEventListener("click", function () {
        resultSection.classList.add("hidden");
        resultFound.classList.add("hidden");
        resultNotFound.classList.add("hidden");
        resultAdded.classList.add("hidden");
        dropZoneContent.classList.remove("hidden");
        previewContainer.classList.add("hidden");
        btnRecognize.disabled = true;
        currentFile = null;
        currentBase64 = null;
        fileInput.value = "";
        document.getElementById("new-title").value = "";
        document.getElementById("result-confidence-bar").style.width = "0%";
    });

});
