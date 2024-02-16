document.addEventListener('DOMContentLoaded', function () {
    // Wait for 5 seconds after the page is fully loaded before showing the modal
    setTimeout(function () {
        var myModal = new bootstrap.Modal(document.getElementById('myModal'), {
            keyboard: false,
            backdrop: 'static'
        });

        myModal.show();
    }, 30000); // 5000 milliseconds = 5 seconds
});


document.addEventListener('DOMContentLoaded', function() {
    setTimeout(function() {
            var hiddenButton = document.getElementById('hidden_but');
            if (hiddenButton) {
                hiddenButton.style.display = 'block'; // Make the element visible
            }
        }
    , 30000);
});

document.addEventListener('DOMContentLoaded', function() {
    fetch('https://pace-risk-map-x35exdywcq-ue.a.run.app/', {
        mode: 'no-cors' // This mode allows the request to be made but does not allow access to the response.
    })
    },);

