
//Returns a random integer between 0 and the given value
function getRandomInt(max) {
    return Math.floor(Math.random() * Math.floor(max));
}

//Once input is given, display random result on page
$("#moment-pane-file-btn")[0].oninput = function() {

    //append the file to the formdata
    var data=new FormData()
    data.append('audio',$("#moment-pane-file-btn")[0].files[0])

    //show the file
    for (var p of data) {
        console.log(p);
    }
    // Pass the audio file to the server by calling the upload method
    // Display the resulting label upon completion
    $.ajax({
        url:"/upload",
        type:'POST',
        data: new FormData($("#moment-pane-upload-form")[0]),
        cache: false,
        processData: false,
        contentType: false,
        error: function(){
            console.log("upload error")
        },
        success: function(label){
            $("#resultText").html("We predict that you are from the <b>" + label + "</b> region.");
        }
    })

    //Scroll the page to result region
    $('html,body').animate({
            scrollTop: $("#results").offset().top},
        'slow');

}


$("#submitButton").click(function() {
    $("#submitMessage").html("<b>Your submission has been received, thank you!</b>");
});


$.get( "ajax/index.html", function( data ) {
    $( ".result" ).html( data );
    alert( "Load was performed." );
});