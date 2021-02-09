// path.onClick = () => {
//     document.getElementById('modalToggle').click();
//     let eggFlaggerDialog = document.getElementById('egg-flagger-dialog');
//     let embeddedImg = document.getElementById('embedded-img')
//     let bboxIndex = eggLayingRegions[path.id].bboxIndex;
//     let imgName = orderedImgNames[currentImgIndex]
//     let annotation = annotations[imgName][bboxIndex];
//     embeddedImg.setAttribute('hidden', "true")
//     embeddedImg.setAttribute('style',
//         `background-image: url(/uploads/${imgName});` +
//         ` height: ${annotation.bbox[3]}px; ` +
//         ` width: ${annotation.bbox[2]}px;` +
//         ' background-origin: content-box;' +
//         `background-position: -${annotation.bbox[0]}px -${annotation.bbox[1]}px;`);
//     let embeddedImgToggleBtn = document.getElementById('flagged-img-toggle');
//     embeddedImgToggleBtn.innerText = 'Show flagged image'
//     embeddedImgToggleBtn.onclick = () => {
//         if (embeddedImg.hasAttribute('hidden')) {
//             embeddedImg.removeAttribute('hidden');
//             embeddedImgToggleBtn.innerText = 'Hide flagged image';
//         } else {
//             embeddedImg.setAttribute('hidden', 'true');
//             embeddedImgToggleBtn.innerText = 'Show flagged image';
//         }
//     }
// }

/*
and the HTML
<div class="modal" id="modal-1" aria-hidden="true">
        <div tabindex="-1" data-micromodal-close>
            <div id='egg-flagger-div' role="dialog" aria-modal="true" aria-labelledby="modal-1-title">
                <header>
                    <h2 id="modal-1-title">
                        Submit error report
                    </h2>
                    <div id="close-button">
                        <div class="close-button-stroke1">
                            <div class="close-button-stroke2"></div>
                        </div>
                    </div>
                    <button id=close-modal hidden aria-label="Close modal" data-micromodal-close></button>
                </header>
                <div id="egg-flagger-dialog">
                    <form>
                        <button id=add-numeric-input type="button">I want to submit both the image and my estimate of
                            the egg count</button>
                        <button id=skip-numeric-input type="button">I want to submit the image only, <strong>no</strong>
                            estimate.</button>
                        <button id=flagged-img-toggle type="button">Show flagged image</button>
                    </form>
                    <div hidden id=egg-count-editor>
                        <form>
                            <label for="flagged-img-egg-count">My egg count:</label>
                            <input id=flagged-img-egg-count type="text" placeholder="e.g., 42">
                            <button disabled type="button" id="submit-error-with-count">Submit</button>
                        </form>
                    </div>
                    <div id=embedded-img></div>
                </div>
            </div>
        </div>
    </div>
*/