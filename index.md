<style>
@media screen and (min-width: 64em) {
.main-content {
    max-width: 70rem;
}
}
.page-header{
background-color: #086375;
background-image: linear-gradient(120deg, #9E5F7D, #A67794);
}
.erp_image {
    width: 12.4rem;
    object-fit: fill;
}
.main-content table th, .main-content table td {
    padding: 0;
}
.table_header td {
  text-align: center;
}
.comparison_table {
  border: 1px solid;
}
</style>

<!-- <iframe width="560" height="315" src="https://www.youtube.com/embed/TAK7KavGivo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen style="max-width: 100%; position: relative; left: 50%; transform: translateX(-50%);"></iframe> -->

## Abstract

[![Teaser image of Continuous Levels of Detail for Light Field Networks](resources/teaser.jpg)](#)

Recently, several approaches have emerged for generating neural representations with multiple levels of detail (LODs). LODs can improve the rendering by using lower resolutions and smaller model sizes when appropriate. However, existing methods generally focus on a few discrete LODs which suffer from aliasing and flicker artifacts as details are changed and limit their granularity for adapting to resource limitations. In this paper, we propose a method to encode light field networks with continuous LODs, allowing for finely tuned adaptations to rendering conditions. Our training procedure uses summed-area table filtering allowing efficient and continuous filtering at various LODs. Furthermore, we use saliency-based importance sampling which enables our light field networks to distribute their capacity, particularly limited at lower LODs, towards representing the details viewers are most likely to focus on. Incorporating continuous LODs into neural representations enables progressive streaming of neural representations, decreasing the latency and resource utilization for rendering.

## Downloads
(coming soon)

<!-- <div style="display: flex; text-align:center; flex-direction: row; flex-wrap: wrap;">
<div style="margin:1rem; flex-grow: 1;"><a href="#"><img style="max-width:10rem;" src="resources/paper.jpg"><br>Paper</a><br></div>
<div style="margin:1rem; flex-grow: 1;"><a href="resources/supplementary.pdf"><img style="max-width:10rem;" src="resources/supplementary.jpg"><br>Supplementary</a></div>
<div style="margin:1rem; flex-grow: 1;"><a href="https://github.com/AugmentariumLab/multiscale-lfn"><img style="max-width:10rem;" src="resources/github.jpg"><br>Code</a></div>
</div> -->

<!-- 
## Rendered Examples

<div class='container'>
  <a href="multiple_lfn_comparison.html" rel="noopener noreferrer">See also Multiple LFNs vs Multi-scale LFNs</a>
  <br>
  <table class="comparison_table" cellspacing="3">
    <tr class="table_header">
      <td>
        Model
      </td>
      <td>
        LOD 1 (1/8 scale)
      </td>
      <td>
        LOD 2 (1/4 scale)
      </td>
      <td>
        LOD 3 (1/2 scale)
      </td>
      <td>
        LOD 4 (1/1 scale)
      </td>
    </tr>
    <tr>
      <td>Single-scale LFN</td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset1/fullscale_r8.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset1/fullscale_r4.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset1/fullscale_r2.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset1/fullscale_r1.mp4" type="video/mp4">
      </video></td>
    </tr>
    <tr>
      <td>Multi-scale LFN</td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset1/multiscale_r8.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset1/multiscale_r4.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset1/multiscale_r2.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset1/multiscale_r1.mp4" type="video/mp4">
      </video></td>
    </tr>
    <tr>
      <td>Single-scale LFN</td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset2/fullscale_r8.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset2/fullscale_r4.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset2/fullscale_r2.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset2/fullscale_r1.mp4" type="video/mp4">
      </video></td>
    </tr>
    <tr>
      <td>Multi-scale LFN</td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset2/multiscale_r8.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset2/multiscale_r4.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset2/multiscale_r2.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset2/multiscale_r1.mp4" type="video/mp4">
      </video></td>
    </tr>
    <tr>
      <td>Single-scale LFN</td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset3/fullscale_r8.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset3/fullscale_r4.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset3/fullscale_r2.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset3/fullscale_r1.mp4" type="video/mp4">
      </video></td>
    </tr>
    <tr>
      <td>Multi-scale LFN</td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset3/multiscale_r8.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset3/multiscale_r4.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset3/multiscale_r2.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset3/multiscale_r1.mp4" type="video/mp4">
      </video></td>
    </tr>
    <tr>
      <td>Single-scale LFN</td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset4/fullscale_r8.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset4/fullscale_r4.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset4/fullscale_r2.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset4/fullscale_r1.mp4" type="video/mp4">
      </video></td>
    </tr>
    <tr>
      <td>Multi-scale LFN</td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset4/multiscale_r8.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset4/multiscale_r4.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset4/multiscale_r2.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset4/multiscale_r1.mp4" type="video/mp4">
      </video></td>
    </tr>
    <tr>
      <td>Single-scale LFN</td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset5/fullscale_r8.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset5/fullscale_r4.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset5/fullscale_r2.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset5/fullscale_r1.mp4" type="video/mp4">
      </video></td>
    </tr>
    <tr>
      <td>Multi-scale LFN</td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset5/multiscale_r8.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset5/multiscale_r4.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset5/multiscale_r2.mp4" type="video/mp4">
      </video></td>
      <td><video muted autoplay loop class="erp_image">
        <source src="resources/videos/dataset5/multiscale_r1.mp4" type="video/mp4">
      </video></td>
    </tr>
  </table>
</div> -->

## Citation
(coming soon)

<!-- ```bibtex
@inproceedings{li2022progressive,
  author={Li, David and Varshney, Amitabh},
  booktitle={2022 International Conference on 3D Vision (3DV)}, 
  title={Progressive Multi-Scale Light Field Networks}, 
  year={2022},
  volume={},
  number={},
  pages={231-241},
  doi={10.1109/3DV57658.2022.00035}
}
```

David Li, and Amitabh Varshney. Progressive Multi-Scale Light Field Networks. In 2022 International Conference on 3D Vision (3DV), Prague, Czech Republic, 2022, pp. 231-241, doi: 10.1109/3DV57658.2022.00035. -->
