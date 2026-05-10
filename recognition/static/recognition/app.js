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
    const topkSection = document.getElementById("topk-section");
    const topkList = document.getElementById("topk-list");
    const resultLowConfidence = document.getElementById("result-low-confidence");
    const resultAdded = document.getElementById("result-added");
    const addIdentityForm = document.getElementById("add-identity-form");
    const btnReset = document.getElementById("btn-reset");
    const webcamModal = document.getElementById("webcam-modal");
    const webcamVideo = document.getElementById("webcam-video");
    const webcamCanvas = document.getElementById("webcam-canvas");
    const btnWebcamClose = document.getElementById("btn-webcam-close");
    const btnWebcamCapture = document.getElementById("btn-webcam-capture");

    const CONFIDENCE_THRESHOLD = 70;

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
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "user", width: 640, height: 480 }
            });
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
        topkSection.classList.add("hidden");
        resultLowConfidence.classList.add("hidden");
        resultAdded.classList.add("hidden");

        const confidence = data.prediction.confidence;
        const isHighConfidence = confidence >= CONFIDENCE_THRESHOLD;

        if (data.prediction && data.prediction.class_id) {
            resultFound.classList.remove("hidden");

            const avatarEl = document.getElementById("result-avatar");
            if (data.identity && data.identity.avatar_url) {
                avatarEl.src = data.identity.avatar_url;
                avatarEl.parentElement.classList.remove("hidden");
            } else {
                avatarEl.parentElement.classList.add("hidden");
            }

            const titleEl = document.getElementById("result-title");
            titleEl.textContent = (data.identity && data.identity.title)
                ? data.identity.title
                : data.prediction.class_id;

            document.getElementById("result-class").querySelector("span").textContent = data.prediction.class_id;
            document.getElementById("result-confidence-text").textContent = confidence + "%";
            setTimeout(function () {
                document.getElementById("result-confidence-bar").style.width = confidence + "%";
            }, 100);
        }

        if (!isHighConfidence) {
            resultLowConfidence.classList.remove("hidden");

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

        if (data.topk && data.topk.length > 0) {
            topkSection.classList.remove("hidden");
            topkList.innerHTML = "";

            data.topk.forEach(function (item) {
                const row = document.createElement("div");
                row.className = "topk-row flex items-center gap-3 py-3 px-4 rounded-xl";

                const colors = ["bg-violet-500/20 text-violet-300", "bg-blue-500/15 text-blue-300", "bg-slate-500/15 text-slate-300", "bg-slate-500/10 text-slate-400", "bg-slate-500/10 text-slate-500"];
                const colorClass = colors[Math.min(item.rank - 1, colors.length - 1)];

                const name = item.title || item.class_id;

                var avatarHtml;
                if (item.avatar_url) {
                    avatarHtml = '<img src="' + item.avatar_url + '" class="w-9 h-9 rounded-full object-cover shrink-0" alt="">';
                } else {
                    avatarHtml = '<div class="w-9 h-9 rounded-full bg-slate-700/60 flex items-center justify-center shrink-0"><i data-lucide="user" class="w-4 h-4 text-slate-500"></i></div>';
                }

                row.innerHTML =
                    '<div class="rank-badge ' + colorClass + '">' + item.rank + '</div>' +
                    avatarHtml +
                    '<div class="flex-1 min-w-0">' +
                        '<p class="text-sm font-medium text-slate-200 truncate">' + name + '</p>' +
                        '<p class="text-xs text-slate-500 font-mono">' + item.class_id + '</p>' +
                    '</div>' +
                    '<div class="text-right shrink-0">' +
                        '<p class="text-sm font-semibold text-violet-400">' + item.confidence + '%</p>' +
                        '<p class="text-xs text-slate-500">cos ' + item.cosine_similarity + '</p>' +
                    '</div>';

                topkList.appendChild(row);
            });
        }

        resultSection.classList.add("slide-up");
        uploadSection.style.display = "";
        lucide.createIcons();
    }

    function generateClassId() {
        return "id_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 6);
    }

    addIdentityForm.addEventListener("submit", async function (e) {
        e.preventDefault();
        const title = document.getElementById("new-title").value.trim();

        if (!title) return;

        const classId = generateClassId();

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
                resultLowConfidence.classList.add("hidden");
                resultAdded.classList.remove("hidden");
                document.getElementById("added-message").textContent =
                    '"' + data.identity.title + '" has been added and will be recognized in future identifications.';
                lucide.createIcons();
            } else if (data.error) {
                alert(data.error);
            }
        } catch (err) {}
    });

    btnReset.addEventListener("click", function () {
        resultSection.classList.add("hidden");
        resultFound.classList.add("hidden");
        topkSection.classList.add("hidden");
        resultLowConfidence.classList.add("hidden");
        resultAdded.classList.add("hidden");
        dropZoneContent.classList.remove("hidden");
        previewContainer.classList.add("hidden");
        btnRecognize.disabled = true;
        currentFile = null;
        currentBase64 = null;
        fileInput.value = "";
        document.getElementById("new-title").value = "";
        document.getElementById("result-confidence-bar").style.width = "0%";
        topkList.innerHTML = "";
    });

});
