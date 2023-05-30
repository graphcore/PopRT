// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"use strict";

// Colour the cells of a table as a heatmap of values. Intended for use
// with HTML generated from RST code by Sphinx (but could be used with
// any HTML).
//
// In the simplest use case, just add the "heatmap" class to a table and
// make a call to the function in your rst source:
//
//   .. table:: Caption goes here
//      :class: heatmap
//
//      ... table definition
//
//   .. raw:: html
//
//       <script>heatmapify()</script>
//
// This will apply the heatmap to all tables with the class "heatmap"
// using the default colours, etc.
//
// There are a few parameters you can use to customize the effect.
// You can use different class names to highlight different tables
// in different ways:
//       <script>
//         // Change default class to red
//         heatmapify({rgb: "255,0,0"})
//         // Another class with inverted scale in cyan and highlighting of lowest value
//         heatmapify({tclass: "coldmap", invert: true, rgb: "0,255,255", highlight_min: true})
//       </script>
//
// You can also use custom css to apply whatever other styling you want to the table class.
//
function heatmapify(
    {
        showValues = true,        // Set to false to hide values and just show colours
        rgb = "255,128,64",       // Basic cell colour to use - adjusted with a transparency (alpha) value
        rgb_map = null,           // A list of RGB values to use instead of rgb parameter
        tclass = "heatmap",       // CSS class to look for
        align_numbers = "right",  // Alignment for numerical cells
        align_text = "left",      // Alignment for text in cells
        highlight_min = false,    // Highlight (bold) the minimum value
        highlight_max = false,    // Highlight (bold) the maximum value
        invert = false,           // Strongest colour for the smallest value
        thousands = false,        // Add a thousands separator to the numbers
        quantise = 0,             // Number of levels of colour (0 = no limit)
    } = {}
) {
    $(document).ready(function () {
        var min, max;
        const values = [];
        // Get the min and max values in order to normalise the cell values
        $("table." + tclass + " td").each(function () {
            let val = $(this).text();
            if (Boolean(val) && !isNaN(val)) {
                values.push(val);
            }
        })
        min = Math.min.apply(null, values);
        max = Math.max.apply(null, values);
        // Format the table cells
        $("table." + tclass + " td > p:first-child").each(function () {
            let val = $(this).text();
            if (Boolean(val) && !isNaN(val)) {
                // Generate a background for each cell with colour proportional to the value
                let a = (val - min) / (max - min);
                if (invert)
                    a = 1.0 - a;
                // Make levels quantised if needed
                if (rgb_map)
                    quantise = rgb_map.length - 1
                if (quantise)
                    a = Math.floor(a * quantise) / quantise
                if (rgb_map) {
                    // Pick a colour from the rgb map
                    i = Math.round(a * (rgb_map.length-1))
                    rgb = rgb_map[i]
                    // Set the style for the cell
                    $(this).parent().css({
                        "background-color": "rgb(" + rgb + ")",
                        "color": "#292C31",
                        "text-align": align_numbers
                    });
                } else {
                    // Make sure lowest level has some shading
                    a += 0.05
                    // Set the style for the cell
                    $(this).parent().css({
                        "background-color": "rgba(" + rgb + "," + a + ")",
                        "color": "#292C31",
                        "text-align": align_numbers
                    });
                }
                // Optionally hide the text
                if (!showValues)
                    $(this).css({ "display": "none" })
                // If (approx) equal to min or max value, then highlight it
                if (highlight_min && val <= Math.ceil(min)) {
                    $(this).css({ "font-weight": "bold" });
                }
                if (highlight_max && val >= Math.floor(max)) {
                    $(this).css({ "font-weight": "bold" });
                }
                // Set the thousands separator
                if (thousands)
                    $(this).text(val.toString().replace(/(\d)(?=(\d{3})+(?!\d))/g, '$1,'))
            } else {
                // It is a non-numeric cell so set the default background
                // to get rid of alternate row highlighting
                $(this).css({ "background-color": "inherit", "text-align": align_text });
            }
        });
    });
}

// Update various document elements once the page has loaded
function updateDocumentLinks() {
    // Add onclick handler to boxes containing doc links
    let doc_links = document.querySelectorAll("div.flex-box")
    for (let i = 0; i < doc_links.length; i++) {
        let box = doc_links[i];
        box.onclick = function() {document.location = this.firstElementChild.firstElementChild.href};
        box.style.cursor = "pointer";
    }
    // Change external links to open in new window/tab
    for (let i = 0; i < document.links.length; i++) {
        if (document.links[i].classList.contains("internal")) continue;
        if (document.links[i].protocol == "mailto:") continue;
        if (document.links[i].hostname.includes("github.com")) {
            document.links[i].target = "_blank";
            document.links[i].title = "Link to GitHub repository";
            document.links[i].classList.add("fa-after");
            document.links[i].classList.add("fa-github-after");
        }
        else if ((document.location.hostname != document.links[i].hostname) && (document.links[i].hostname != "docs.graphcore.ai")) {
            document.links[i].setAttribute("target", "_blank");
            document.links[i].title = "External link ("+document.links[i].hostname+")";
            document.links[i].classList.add("fa-after");
            document.links[i].classList.add("fa-arrow-up-right-from-square-after");
        }
    }
}

function createSelector(links) {
    if (links) {
        var el = undefined
        const selector = document.createElement("select");
        for (let j = 0; j < links.length; j++) {
            let opt = document.createElement('option');
            opt.value = links[j].href;
            var text = links[j].innerText;
            if (document.location.pathname.includes("/"+text+"/"))
                opt.selected = true;
            if (text == "zh_CN") text = "中文"
            if (text == "en") text = "English"
            opt.innerText = text
            selector.appendChild(opt);
        }
        if (selector.options.length > 1) {
            selector.className="toctree-l1";
            selector.onchange = function(){window.location = selector.value};
            el = selector
        }
    }
    return el
}

const mutationObserver = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        if (!mutation.addedNodes) return

        // Look for content added by Read the Docs and change it for usability
        const selectors = document.createElement("li");
        var pdf_link = undefined
        selectors.className="toctree-l1";
        for (let i = 0; i < mutation.addedNodes.length; i++) {
            let node = mutation.addedNodes[i];
            if (node.nodeName == "DIV" && node.classList == "injected") {
                const newList = document.createElement("ul");
                newList.style.borderBottom = "thin solid #eee";
                newList.style.paddingBottom = "12px";
                newList.style.marginBottom = "12px";

                for (const dt of node.querySelectorAll("dl > dt")) {
                    // Set the nofollow and download attributes for PDF and move into the new menu
                    if (dt.innerText == "Downloads") {
                        pdf_link = dt.parentNode.querySelector("dd > a")
                        if (pdf_link && pdf_link.innerText == "PDF") {
                            pdf_link.setAttribute("rel", "nofollow");
                            pdf_link.setAttribute("download", "");
                            pdf_link.className="reference internal";
                            pdf_link.innerHTML = '<span class="fa fa-file-pdf-o fa-file-pdf">&nbsp;&nbsp;</span>Download PDF';
                        }
                    }
                    // Populate the language selector
                    if (dt.innerText == "Languages"){
                        const sel = createSelector(dt.parentNode.querySelectorAll("dd > a"));
                        if (sel) selectors.append(sel)
                    }
                    // populate the version selector
                    if (dt.innerText == "Versions") {
                        const sel = createSelector(dt.parentNode.querySelectorAll("dd > a"));
                        if (sel) selectors.append(sel)
                    }
                }
                if (selectors.childElementCount > 0) {
                    const p = document.createElement("p");
                    p.append(selectors)
                    newList.append(p)
                }
                if (pdf_link) {
                    const li = document.createElement("li");
                    newList.append(li);
                    li.className="toctree-l1";
                    li.append(pdf_link);
                }
                if (newList.childElementCount > 0) {
                    const menu = document.querySelector("div.wy-menu.wy-menu-vertical")
                    menu.prepend(newList)
                }
                document.querySelector("div.rst-versions").style.display = "none";
            }
        }
    })
})
