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
$("#moment-pane-file-btn")[0].oninput = function() {

    
    var data=new FormData()
          data.append('audio',$("#moment-pane-file-btn").files)

    for (var p of data) {
      console.log(p);
    }
          $.ajax({
              url:"/upload",
              type:'POST',
              data: data,//new FormData($("#moment-pane-upload-form")[0]),
              cache: false,
              processData: false,
              contentType: false,
              error: function(){
                  console.log("upload error")
              },
              success: function(data){
                  console.log(data);
                  console.log("upload success");
              }
          })


    //Generate random region
    var randomIndex = getRandomInt(6);
    var region = regions[randomIndex];
    
    //Display result
    $("#resultText").html("We predict that you are from the <b>" + region + "</b> region.");

    //Scroll the page to result region
    $('html,body').animate({
        scrollTop: $("#results").offset().top},
        'slow');
        
  }


$("#submitButton").click(function() {
    $("#submitMessage").html("<b>Your submission has been received, thank you!</b>");
});