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