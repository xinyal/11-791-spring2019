//A dictionary mapping numbers to our classification regions
var regions = {
    0: "Mid-Atlantic",
    1: "Midland",
    2: "New England",
    3: "Northern",
    4: "Southern",
    5: "Western"
}

//Returns a random integer between 0 and the given value
function getRandomInt(max) {
    return Math.floor(Math.random() * Math.floor(max));
}

//Once input is given, display random result on page
$("#fileInput")[0].oninput = function() {
    //Generate random region
    var randomIndex = getRandomInt(6);
    var region = regions[randomIndex];

    //Display result
    $("#resultText").html("We predict that you are from the <b>" + region + "</b> region.");

    //Scroll the page to result region
    $('html,body').animate({
        scrollTop: $("#results").offset().top},
        'slow');
};


$("#submitButton").click(function() {
    $("#submitMessage").html("<b>Your submission has been received, thank you!</b>");
});