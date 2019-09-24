var qReqs = 0;
var reqsPage, modelsPage, catsPage, docsPage;
var activePage = null;
var activeId = 0;
var builtIds = [false, false, false, false];
var fullData = JSON.parse(fullInfo);
var docsDict = JSON.parse(docsInfo);
var reqs = {};
var pages = ["#pageRequests", "#pageModels", "#pageCategories", "#pageDocuments"];

function buildPages() {
    let main = document.querySelector('#mainPage');
    reqsPage = document.createElement('div');
    reqsPage.id = 'reqsPage';
    reqsPage.className = 'tabbed';
    main.appendChild(reqsPage);
    modelsPage = document.createElement('div');
    modelsPage.id = 'modelsPage';
    modelsPage.className = 'tabbed';
    main.appendChild(modelsPage);
    catsPage = document.createElement('div');
    catsPage.id = 'catsPage';
    catsPage.className = 'tabbed';
    main.appendChild(catsPage);
    docsPage = document.createElement('div');
    docsPage.id = 'docsPage';
    docsPage.className = 'tabbed';
    main.appendChild(docsPage);
}

function addRequest(node, id) {
    if (node.checked) {
        qReqs++;
        reqs[id] = true
    }
    else if(qReqs) {
        qReqs--;
        delete reqs[id]
    }
    else {
        return;
    }
    if (!qReqs) {
        buildActivePage();
        //activePage = null;
        return;
    }
    if (!activePage) {
        buildPages();
        activePage = reqsPage;
        activeId = 1;
        document.querySelector(pages[activeId-1]).classList.add("active");
    }
    for (i=0; i<builtIds.length; i++)
        builtIds[i] = false;
    buildActivePage();
}

function addAllRequests(node) {
    let chks = document.querySelectorAll(".chk");
    let needSelect = node.innerHTML == "Select All";
    chks.forEach(function(chk) {
        if (needSelect && !chk.checked || !needSelect && chk.checked)
       chk.click();
    });
    if (node.innerHTML == "Select All")
        node.innerHTML = "Unselect All";
    else
        node.innerHTML = "Select All";
}

function buildActivePage() {
    switch (activeId) {
        case 1:
            buildReqsPage();
            break;
        case 2:
            buildModelsPage();
            break;
        case 3:
            buildCatsPage();
            break;
        case 4:
            buildDocsPage()
        default:
            break;
    }
    builtIds[activeId-1] = true;
    activePage.classList.add("vis");
}

function changePage(pageId) {
    if (!qReqs) {
        return;
    }
    activePage.classList.remove('vis');
    document.querySelector(pages[activeId-1]).classList.remove("active");
    if (pageId == "Requests") {
        activePage = reqsPage;
        activeId = 1;
    }
    else if(pageId == "Models") {
        activePage = modelsPage;
        activeId = 2;
    }
    else if(pageId == "Categories") {
        activePage = catsPage;
        activeId = 3;
    }
    else {
        activePage = docsPage;
        activeId = 4;
    }
    document.querySelector(pages[activeId-1]).classList.add("active");
    if (!builtIds[activeId-1])
        buildActivePage(reqs)
    activePage.classList.add("vis");
}

function buildReqsPage() {
    while(reqsPage.firstChild) {
        reqsPage.removeChild(reqsPage.firstChild);
    }
    if (!qReqs)
        return;
    let tabCols = Object.keys(reqs).length;
    let table = document.createElement("table");
    table.className = "commontable";
    reqsPage.appendChild(table);
    let trh = document.createElement("tr");
    table.appendChild(trh);
    let ch = document.createElement("th");
    ch.innerHTML = "Info";
    ch.className = "features";
    trh.appendChild(ch);
    let keys = Object.keys(reqs);
    for (key in reqs) {
        ch = document.createElement("th");
        trh.appendChild(ch);
        let title = document.createTextNode("Request " + key);
        ch.appendChild(title);
    }
    let tr = document.createElement("tr");
    table.appendChild(tr);
    ch = document.createElement("td");
    ch.className = "bold";
    ch.innerHTML = "Data Set";
    tr.appendChild(ch);
    for (key in reqs) {
        ch = document.createElement("td");
        ch.innerHTML = fullData[key]["datasetPath"];
        tr.appendChild(ch);
    }
    tr = document.createElement("tr");
    table.appendChild(tr);
    ch = document.createElement("td");
    ch.className = "bold";
    ch.innerHTML = "Sources";
    tr.appendChild(ch);
    for (key in reqs) {
        ch = document.createElement("td");
        ch.innerHTML = fullData[key].sourcesPath;
        tr.appendChild(ch);
    }
    tr = document.createElement("tr");
    table.appendChild(tr);
    ch = document.createElement("td");
    ch.innerHTML = "Preprocess";
    ch.className = "expanded bold"
    ch.colspan = keys.length + 1;
    tr.appendChild(ch);
    tr = document.createElement("tr");
    table.appendChild(tr);
    ch = document.createElement("td");
    ch.innerHTML = "Preprocess";
    ch.className = "expandedcont bold";
    tr.appendChild(ch);
    for (key in reqs) {
        ch = document.createElement("td");
        ch.innerHTML = fullData[key]["preprocess"]["language_tokenization"];
        tr.appendChild(ch);
    }
    tr = document.createElement("tr");
    table.appendChild(tr);
    ch = document.createElement("td");
    ch.innerHTML = "Normalization";
    ch.className = "expandedcont bold";
    tr.appendChild(ch);
    for (key in reqs) {
        ch = document.createElement("td");
        ch.innerHTML = fullData[key]["preprocess"]["normalization"];
        tr.appendChild(ch);
    }
    tr = document.createElement("tr");
    table.appendChild(tr);
    ch = document.createElement("td");
    ch.innerHTML = "Exclude stop words";
    ch.className = "expandedcont bold";
    tr.appendChild(ch);
    for (key in reqs) {
        ch = document.createElement("td");
        ch.innerHTML = fullData[key]["preprocess"]["stop_words"];
        tr.appendChild(ch);
    }
    tr = document.createElement("tr");
    table.appendChild(tr);
    ch = document.createElement("td");
    ch.innerHTML = "Excluded POS";
    ch.className = "expandedcont bold";
    tr.appendChild(ch);
    for (key in reqs) {
        ch = document.createElement("td");
        ch.innerHTML = fullData[key]["preprocess"]["exclude_positions"] || "none";
        tr.appendChild(ch);
    }
    tr = document.createElement("tr");
    table.appendChild(tr);
    ch = document.createElement("td");
    ch.innerHTML = "Excluded words";
    ch.className = "expandedcont bold";
    tr.appendChild(ch);
    for (key in reqs) {
        ch = document.createElement("td");
        ch.innerHTML = fullData[key]["preprocess"]["extra_words"] || "none";
        tr.appendChild(ch);
    }
    tr = document.createElement("tr");
    table.appendChild(tr);
    ch = document.createElement("td");
    ch.innerHTML = "Categories";
    ch.className = "expanded bold"
    ch.colspan = keys.length + 1;
    tr.appendChild(ch);
    tr = document.createElement("tr");
    table.appendChild(tr);
    ch = document.createElement("td");
    ch.innerHTML = "At all";
    ch.className = "expandedcont bold";
    tr.appendChild(ch);
    for (key in reqs) {
        ch = document.createElement("td");
        ch.innerHTML = fullData[key]["categories"].length;
        tr.appendChild(ch);
    }
    tr = document.createElement("tr");
    table.appendChild(tr);
    ch = document.createElement("td");
    ch.innerHTML = "Excluded categories";
    ch.className = "expandedcont bold";
    tr.appendChild(ch);
    for (key in reqs) {
        ch = document.createElement("td");
        ch.innerHTML = fullData[key]["preprocess"]["excat"] || "none";
        tr.appendChild(ch);
    }
    tr = document.createElement("tr");
    table.appendChild(tr);
    ch = document.createElement("td");
    ch.innerHTML = "Documents";
    ch.className = "expanded bold";
    ch.colspan = keys.length + 1;
    tr.appendChild(ch);
    tr = document.createElement("tr");
    table.appendChild(tr);
    ch = document.createElement("td");
    ch.innerHTML = "At all";
    ch.className = "expandedcont bold";
    tr.appendChild(ch);
    for (key in reqs) {
        ch = document.createElement("td");
        ch.innerHTML = Object.keys(fullData[key]["docs"]).length;
        tr.appendChild(ch);
    }
    tr = document.createElement("tr");
    table.appendChild(tr);
    ch = document.createElement("td");
    ch.innerHTML = "Labels";
    ch.className = "expandedcont bold";
    tr.appendChild(ch);
    for (key in reqs) {
        ch = document.createElement("td");
        let labs = 0;
        for (doc in fullData[key]["docs"]) {
            labs += fullData[key]["docs"][doc]["actual"].split(",").length;
        }
        ch.innerHTML = "" + labs;
        tr.appendChild(ch);
    }
    tr = document.createElement("tr");
    table.appendChild(tr);
    ch = document.createElement("td");
    ch.innerHTML = "Models (F1-Measure)";
    ch.className = "expanded bold"
    ch.colspan = keys.length + 1;
    tr.appendChild(ch);

    let qModels = 0;
    for (key in reqs) {
        if (Object.keys(fullData[key]["models"]).length > qModels)
            qModels = Object.keys(fullData[key]["models"]).length;
    }
    let f1 = 0;
    for (i=0; i<qModels; i++) {
        tr = document.createElement("tr");
        ch = document.createElement("td");
        tr.append(ch);
        table.appendChild(tr);
        for (key in reqs) {
            ch = document.createElement("td");
            models = Object.keys(fullData[key]["models"]);
            if (i < models.length) {
                if (fullData[key]["models"][models[i]]["all"]["f1"] > f1) {
                    let elf1 = document.querySelectorAll("span.f1");
                    elf1.forEach(function(sp) {
                        sp.classList.remove("f1");
                    });
                    f1 = fullData[key]["models"][models[i]]["all"]["f1"];
                }
                let spanclass = "";
                if (fullData[key]["models"][models[i]]["all"]["f1"] == f1)
                    spanclass = "f1";
                ch.innerHTML = "<span class='" + spanclass + "'>" + models[i].toUpperCase() + " (" +
                    (fullData[key]["models"][models[i]]["all"]["f1"] *100).toFixed(2) + "%)</span>";
            }
            else
                ch.innerHTML = "";
            tr.appendChild(ch);
        }
    }
}

function buildModelsPage() {
    while(modelsPage.firstChild) {
        modelsPage.removeChild(modelsPage.firstChild);
    }
    if (!qReqs)
        return;
    let tabCols = Object.keys(reqs).length;
    let table = document.createElement("table");
    table.className = "commontable";
    modelsPage.appendChild(table);
    let trh = document.createElement("tr");
    table.appendChild(trh);
    let ch = document.createElement("th");
    ch.innerHTML = "Info";
    ch.className = "features";
    trh.appendChild(ch);
    let keys = Object.keys(reqs);
    for (key in reqs) {
        for (model in fullData[key]["models"]) {
            ch = document.createElement("th");
            trh.appendChild(ch);
            let title = document.createElement("div");
            title.innerHTML = "Request " + key;
            ch.appendChild(title);
            title = document.createElement("div");
            title.innerHTML = "Model: " + model.toUpperCase()
            ch.appendChild(title);
        }
    }
    tr = document.createElement("tr");
    table.appendChild(tr);
    ch = document.createElement("td");
    ch.innerHTML = "Documents";
    ch.className = "expanded bold";
    ch.colspan = keys.length + 1;
    tr.appendChild(ch);
    for (i=0; i<7; i++) {
        addModelsPageRaw(table, i);
    }
    tr = document.createElement("tr");
    table.appendChild(tr);
    ch = document.createElement("td");
    ch.innerHTML = "Labels";
    ch.className = "expanded bold";
    ch.colspan = keys.length + 1;
    tr.appendChild(ch);
    for (i=7; i<12; i++) {
        addModelsPageRaw(table, i);
    }
    tr = document.createElement("tr");
    table.appendChild(tr);
    ch = document.createElement("td");
    ch.innerHTML = "Metrics";
    ch.className = "expanded bold";
    ch.colspan = keys.length + 1;
    tr.appendChild(ch);
    for (i=12; i<25; i++) {
        addModelsPageRaw(table, i);
    }
    tr = document.createElement("tr");
    table.appendChild(tr);
    ch = document.createElement("td");
    ch.innerHTML = "Categories (by F1 in descending order)";
    ch.className = "expanded bold";
    ch.colspan = keys.length + 1;
    tr.append(ch);
    let cAr = createCatsArrays();
    let rows = 0;
    for (key in cAr) {
        if (cAr[key].length > rows)
            rows = cAr[key].length;
    }
    addModelsPageCatsRows(table, rows, cAr);
}

function addModelsPageRaw(table, ind) {
    let fullNames = ["Documents at all", "Exactly classified", "Classified completely, but with errors",
        "Partially classified", "Classified partially with errors", "Classified falsely", "Not classified",
        "Actual labels", "Pedicted at all", "Predicted correctly", "Predicted falsely", "Not predicted",
        "Exact Match Ratio", "Accuracy", "Precision", "Recall", "F1-Measure", "Hamming Loss",
        "Macro-Averaged Precision", "Macro-Averaged Recall", "Macro-Averaged F1-Measure",
        "Micro-Averaged Precision", "Micro-Averaged Recall", "Micro-Averaged F1-Measure", "Rank threshold"];
    let shortNames= ["d_docs", "dd_ex", "dd_cf", "dd_p", "dd_pf", "dd_f", "dd_n",
                     "d_actual", "d_predicted", "d_correctly", "d_falsely", "d_notPredicted",
                     "emr", "accuracy", "precision", "recall", "f1", "hl", "macroPrecision",
                     "macroRecall", "macroF1", "microPrecision", "microRecall", "microF1", "rank"];
    let tr = document.createElement("tr");
    table.appendChild(tr);
    let ch = document.createElement("td");
    ch.innerHTML = fullNames[ind];
    ch.className = "expandedcont bold";
    tr.appendChild(ch);
    for (key in reqs) {
        for (model in fullData[key]["models"]) {
            ch = document.createElement("td");
            ch.className = "digs";
            tr.appendChild(ch);
             if (shortNames[i] == "rank") {
                let rank = 0.5
                if (fullData[key].hasOwnProperty("ranks"))
                    rank = fullData[key]["ranks"][model];
                if (!rank) {
                    console.log("Rank is " + rank + " for model " + model);
                    rank = 0.5
                }
                ch.innerHTML = (rank * 100).toFixed(2) + "%";
            }
            else if (shortNames[i].startsWith("d"))
                ch.innerHTML = fullData[key]["models"][model]["all"][shortNames[i]];
            else if(shortNames[i].startsWith("hl"))
                ch.innerHTML = fullData[key]["models"][model]["all"][shortNames[i]].toFixed(4);
            else
                ch.innerHTML = (fullData[key]["models"][model]["all"][shortNames[i]] * 100).toFixed(2) + "%";
        }
    }
}

function addModelsPageCatsRows(table, rows, arr) {
    //console.log("Rows: " + rows + ", keys: " + Object.keys(arr).length);
    for (let i=0; i<rows; i++) {
        let tr = document.createElement("tr");
        table.appendChild(tr);
        tr.appendChild(document.createElement("td"));
        for (key in reqs) {
            for (model in fullData[key]["models"]) {
                let keyArr = key + " | " + model;
                //console.log("Key: " + keyArr + ", cats: " + arr[keyArr].length);
                let td = document.createElement("td");
                if (arr[keyArr].length - 1 >= i) {
                    let cat = arr[keyArr][i];
                    //console.dir(cat);
                    td.innerHTML = "<div><table class='catstab'><tr>" +
                        "<td class='txt' style='overflow: hidden; white-space: nowrap; text-overflow: ellipsis;'>" +
                        cat["name"] + "</td>" +
                        "<td class='digs'>" + (cat.f1 * 100).toFixed(2) + "%</td></tr></table></div>"
                }
                tr.appendChild(td);
            }
        }
    }
}

function createCatsArrays() {
    let catsArrays = {}
    for (key in reqs) {
        for (model in fullData[key]["models"]) {
            let arKey = key + " | " + model;
            catsArrays[arKey] = []
            for (category in fullData[key]["models"][model]) {
                if (category != "all") {
                    let catObj = fullData[key]["models"][model][category];
                    catObj["name"] = category;
                    catsArrays[arKey].push(catObj);
                }
            }
            catsArrays[arKey].sort(function(a, b) {
               return b.f1 - a.f1;
            });
        }
    }
    return catsArrays;
}

function buildCatsPage() {
    while(catsPage.firstChild) {
        catsPage.removeChild(catsPage.firstChild);
    }
    if (!qReqs)
        return;
    let tabCols = Object.keys(reqs).length;
    let table = document.createElement("table");
    table.className = "commontable";
    catsPage.appendChild(table);
    let trh = document.createElement("tr");
    table.appendChild(trh);
    let ch = document.createElement("th");
    ch.innerHTML = "Info";
    ch.className = "features";
    trh.appendChild(ch);
    let keys = Object.keys(reqs);
    for (key in reqs) {
        for (model in fullData[key]["models"]) {
            ch = document.createElement("th");
            trh.appendChild(ch);
            let title = document.createElement("div");
            title.innerHTML = "Request " + key;
            ch.appendChild(title);
            title = document.createElement("div");
            title.innerHTML = "Model: " + model.toUpperCase()
            ch.appendChild(title);
        }
    }
    catsObj = {};
    for (key in reqs) {
        let cats = fullData[key]["categories"];
        for (i=0; i<cats.length; i++) {
            if (!catsObj.hasOwnProperty(cats[i]))
                catsObj[cats[i]] = true;
        }
    }
    for (cat in catsObj) {
        addCatsPageRaws(table, cat);
    }
}

function addCatsPageRaws(table, category) {
    let fullNames = ["Labels", "Predicted labels at all", "Correctly predicted labels",
                     "Falsely predicted labels", "Not predicted labels", "Precision", "Recall", "F1-Measure"];
    let shortNames= ["d_actual", "d_predicted", "d_correctly", "d_falsely", "d_notPredicted",
                     "precision", "recall", "f1"];
    let tr = document.createElement("tr");
    table.appendChild(tr);
    let td = document.createElement("td");
    td.className = "expanded bold";
    td.innerHTML = category;
    tr.appendChild(td);
    for (i=0; i<fullNames.length; i++) {
        tr = document.createElement("tr");
        table.appendChild(tr);
        td = document.createElement("td");
        td.className = "expandedcont bold";
        td.innerHTML = fullNames[i];
        tr.appendChild(td);
        for (key in reqs) {
            for (model in fullData[key]["models"]) {
                td = document.createElement("td");
                td.className = "digs";
                tr.appendChild(td);
                if (fullData[key]["models"][model].hasOwnProperty(category)) {
                    if (shortNames[i].startsWith("d"))
                        td.innerHTML = fullData[key]["models"][model][category][shortNames[i]];
                    else
                        td.innerHTML = (fullData[key]["models"][model][category][shortNames[i]]* 100)
                            .toFixed(2) + "%";
                }
                else
                    td.innerHTML = "N/A";
            }
        }
    }
}

function buildDocsPage() {
    while(docsPage.firstChild) {
        docsPage.removeChild(docsPage.firstChild);
    }
    if (!qReqs)
        return;
    let tabCols = Object.keys(reqs).length;
    let table = document.createElement("table");
    table.className = "docstable";
    docsPage.appendChild(table);
    let trh = document.createElement("tr");
    table.appendChild(trh);
    let ch = document.createElement("th");
    ch.innerHTML = "Info";
    ch.className = "features";
    trh.appendChild(ch);
    let keys = Object.keys(reqs);
    for (key in reqs) {
        for (model in fullData[key]["models"]) {
            ch = document.createElement("th");
            trh.appendChild(ch);
            let title = document.createElement("div");
            title.innerHTML = key;
            ch.appendChild(title);
            title = document.createElement("div");
            title.innerHTML = model.toUpperCase()
            ch.appendChild(title);
        }
    }
    for (doc in docsDict) {
        let tr = document.createElement("tr");
        table.appendChild(tr);
        let td = document.createElement("td");
        td.innerHTML = doc;
        td.title = docsDict[doc];
        td.className = "docname";
        tr.appendChild(td);
        let tdClass = "";

        for (key in reqs) {
            for (model in fullData[key]["models"]) {
                td = document.createElement("td");
                tr.append(td);
                if (!fullData[key]["docs"].hasOwnProperty(doc)) {
                    td.innerHTML = "N/A";
                    tdClass = "";
                }
                else {
                    let fullList = fullData[key]["docs"][doc][model].split(" | ");
                    let predList = fullList[0];
                    let remList = "";
                    if (fullList.length > 1)
                        remList = fullList[1];

                    td.title = "Document:  " + doc + "\n" +
                        "Tagged by: " + fullData[key]["docs"][doc]["actual"] + "\n" +
                        //"Predicted:  " + fullData[key]["docs"][doc][model];
                        "Predicted:  " + predList;
                    if (remList != "") {
                        td.title += "\n=== Other: ===";
                        let rems = remList.split(",");
                        let j = 0;
                        for (let i=0; i<rems.length; i++) {
                            if (rems[i] !== "") {
                                j++;
                                if (j%3 == 1)
                                    td.title += "\n";
                                td.title += rems[i];
                                if (i < rems.length - 1)
                                    td.title += ",";
                            }
                        }
                    }
                    let acts = fullData[key]["docs"][doc]["actual"].split(",");
                    //let preds = fullData[key]["docs"][doc][model].split(",");
                    let preds = predList.split(",");
                    let lenPreds = preds.length == 1 && preds[0] == ""? 0 : preds.length;
                    let found = 0;
                    for (let i = 0; i<acts.length; i++) {
                        //if (fullData[key]["docs"][doc][model].indexOf(acts[i]) >=0 )
                        if (predList.indexOf(acts[i]) >=0 )
                            found++;
                    }
                    if (found == acts.length) {
                        if (found == lenPreds)
                            tdClass = "exact";
                        else
                            tdClass = "comp";
                    }
                    else if(found > 0) {
                        if (found == lenPreds)
                            tdClass = "part";
                        else
                            tdClass = "partnerr";
                    }
                    else if(lenPreds > 0)
                        tdClass = "fals";
                    else
                        tdClass = "notpred";
                }
                td.className = tdClass;
            }
        }
    }
}