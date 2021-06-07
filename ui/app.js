Dropzone.autoDiscover = false;
// var toggle_checked = document.getElementById("slider_model").checked;
/* var toggle_checked = $("#slider_model").is(':checked') */
/* console.log(toggle_checked) */

function init() {
    let dz = new Dropzone("#dropzone", {
        url: "/",
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Some Message",
        autoProcessQueue: false
    });

    dz.on("maxfilesexceeded", function (file) {
        alert("You are not allowed to chose more than 1 file!");
        this.removeFile(file);

    });

    dz.on("addedfile", function () {
        $("#resultHolder").hide();
        $("#divClassTable").hide();

        if (dz.files[1] != null) {
            dz.removeFile(dz.files[0]);
            //            $("#resultHolder").hide();
            //            $("#divClassTable").hide();
        }
    });


    dz.on("complete", function (file) {
        let imageData = file.dataURL;
        var togBtn = $("#togBtn").is(':checked')
        console.log(togBtn)
        if (!togBtn)
            var url = "http://127.0.0.1:5000/img_classify_log_reg";         // this url for logistic_regression model
        else var url = "http://127.0.0.1:5000/img_classify_svm";    // this url for svm model

        $.post(url, {
            image_data: file.dataURL
        }, function (data, status) {

            console.log(data);
            if (!data || data.length == 0) {
                $("#resultHolder").hide();
                $("#divClassTable").hide();
                $("#error").show();
                return;
            }
            let players = ["cristiano_ronaldo", "lionel_messi", "serena_williams", "tom_cruise", "virat_kohli"];

            let match = null;
            let bestScore = -1;
            for (let i = 0; i < data.length; i++) {
                let maxScoreForThisClass = Math.max(...data[i].class_probability);
                // if (maxScoreForThisClass > bestScore) {
                match = data[i];
                bestScore = maxScoreForThisClass;
                // }
            }
            console.log(match)
            console.log(bestScore)
            if (match) {
                $("#error").hide();
                $("#resultHolder").show();
                $("#divClassTable").show();
                $("#resultHolder").html($(`[data-player="${match.class}"`).html());
                let classDictionary = match.class_dictionary;
                for (let personName in classDictionary) {
                    let index = classDictionary[personName];
                    let proabilityScore = match.class_probability[index];
                    let elementName = "#score_" + personName;
                    $(elementName).html(proabilityScore);
                }
            }
            //  dz.removeFile(file);            
        });
    });

    $("#submitBtn").on('click', function (e) {
        dz.processQueue();
    });
}

$(document).ready(function () {
    console.log("ready!");
    $("#error").hide();
    $("#resultHolder").hide();
    $("#divClassTable").hide();

    init();
});