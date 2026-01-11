<font style="color:#000000;">论文链接</font><font style="color:rgb(0, 0, 0);">：</font>[https://arxiv.org/abs/2501.19319](https://arxiv.org/abs/2501.19319)<font style="color:rgb(54, 54, 54);">  
</font><font style="color:#000000;">项目链接：</font>[https://github.com/lastbasket/Endo-2DTAM](https://github.com/lastbasket/Endo-2DTAM)

# 论文内容概述
**摘要**——同步定位与建图（Simultaneous Localization and Mapping，SLAM）在微创手术中的精确外科操作与机器人任务中具有关键作用。<u>尽管近年来三维高斯溅射（3D Gaussian Splatting，3DGS）在新视角合成质量和快速渲染方面显著提升了 SLAM 系统的表现，但由于多视角之间存在不一致性，这类方法在深度与表面重建精度方面仍面临挑战。将 SLAM 与 3DGS 直接结合往往会导致重建帧之间出现不匹配现象</u>。为此，本文提出了一种基于**二维高斯溅射**（2D Gaussian Splatting，2DGS）的实时内窥镜 SLAM 系统——<font style="background-color:#F297CC;">Endo-2DTAM</font>，以应对上述问题。<u>Endo-2DTAM 引入了一种面向表面法向的处理流程，包含跟踪、建图和束调整三个模块，从而实现几何精度更高的重建</u>。所提出的鲁棒跟踪模块结合了点到点（point-to-point）与点到平面（point-to-plane）的距离度量；建图模块则利用法向一致性与深度畸变信息来提升表面重建质量。<u>此外，本文还提出了一种位姿一致的关键帧采样策略，以实现高效且几何一致的关键帧选择</u>。基于公开内窥镜数据集的大量实验结果表明，Endo-2DTAM 在手术场景的深度重建任务中可达到 1.87±0.63 mm 的均方根误差（RMSE），同时保持了计算高效的跟踪性能、高质量的视觉外观以及实时渲染能力。

## INTRODUCTION
内窥镜手术通过实现微创操作，显著降低了患者的术后恢复时间和瘢痕程度，极大地推动了医学领域的发展。该技术使外科医生能够通过微小切口进入体内器官，从而减少对大范围手术干预的需求。<u>然而，人体内部结构复杂且空间狭窄，这为外科医生带来了严峻挑战。内窥镜成像通常具有视场狭窄且缺乏深度感知的问题，这会阻碍医生对复杂三维环境的准确理解与导航能力 。为降低手术风险并提升手术效果，亟需能够提供</u>**<u>实时、高保真三维重建</u>**<u>的可视化工具</u>。

为应对上述问题，同步定位与建图（Simultaneous Localization and Mapping，SLAM）技术被广泛应用于实时估计相机位姿并重建手术场景。<u>传统 SLAM 系统在相机精确跟踪和运行效率方面表现出色，但往往缺乏稠密的几何和纹理信息</u>。例如，广泛使用的 ORB-SLAM3 在高精度相机跟踪和高效运行性能方面取得了显著进展，但其重建依赖于后处理的体素融合过程。尽管近期方法 _[13]–[20]_ 能够通过深度图估计实现稠密重建，但在细粒度几何重建方面仍然存在不足。

<details class="lake-collapse"><summary id="u8c3541e4"><em><span class="ne-text">[13]-[20]</span></em></summary><p id="u292614b4" class="ne-p"><span class="ne-text">[13] </span><strong><span class="ne-text">Vision–Kinematics Interaction for Robotic-Assisted Bronchoscopy Navigation（IEEE Transactions on Medical Imaging 2022）</span></strong><span class="ne-text">：提出视觉信息与机器人运动学模型深度融合的支气管镜导航框架，通过联合优化视觉观测与机械臂先验运动约束，显著提升了复杂支气管结构中位姿估计的稳定性与精度，证明了多源先验约束在机器人辅助手术导航中的重要价值。</span></p><p id="uf706355f" class="ne-p"><span class="ne-text">[14] </span><strong><span class="ne-text">SAGE: SLAM with Appearance and Geometry Prior for Endoscopy（ICRA 2022）</span></strong><span class="ne-text">：提出一种结合外观先验与几何先验的端到端 SLAM 框架，将学习得到的场景外观表征与几何一致性约束引入传统 SLAM 优化过程，在弱纹理、强反光的内窥镜场景中显著提高了定位与重建的鲁棒性，代表了“学习辅助 SLAM”的典型范式。</span></p><p id="u59676288" class="ne-p"><span class="ne-text">[15] </span><strong><span class="ne-text">RNNSLAM: Reconstructing the 3D Colon to Visualize Missing Regions during a Colonoscopy（Medical Image Analysis 2021）</span></strong><span class="ne-text">：利用循环神经网络建模结肠镜视频中的长期时序依赖关系，实现对不可见或遗漏区域的三维结构预测与补全，有效缓解了结肠镜检查中视野受限导致的结构缺失问题，为基于学习的内窥镜三维重建提供了新的时间建模思路。</span></p><p id="u1c9cb741" class="ne-p"><span class="ne-text">[16] </span><strong><span class="ne-text">C³ Fusion: Consistent Contrastive Colon Fusion, Towards Deep SLAM in Colonoscopy（Shape in Medical Imaging Workshop 2023）</span></strong><span class="ne-text">：提出基于对比学习的一致性特征融合策略，将深度特征匹配与时序一致性约束相结合，实现结肠镜视频中的稳健位姿估计与地图融合，推动了从传统几何 SLAM 向深度学习驱动 SLAM 的过渡。</span></p><p id="ud17e5dda" class="ne-p"><span class="ne-text">[17] </span><strong><span class="ne-text">Bimodal Camera Pose Prediction for Endoscopy（IEEE Transactions on Medical Robotics and Bionics 2023）</span></strong><span class="ne-text">：针对内窥镜运动中存在的多解性问题，提出双模态相机位姿预测模型，通过同时建模不同可能运动假设，提高了位姿估计在快速运动和视角剧烈变化条件下的鲁棒性。</span></p><p id="u8178a2dc" class="ne-p"><span class="ne-text">[18] </span><strong><span class="ne-text">EndoDepth-and-Motion: Reconstruction and Tracking in Endoscopic Videos Using Depth Networks and Photometric Constraints（RA-L 2021）</span></strong><span class="ne-text">：将单目深度学习网络与经典光度一致性约束相结合，实现端到端的内窥镜位姿跟踪与三维重建，在无需真实深度标注的情况下获得较为稳定的深度与运动估计，体现了“深度学习 + 几何约束”的混合建模思想。</span></p><p id="u9eb4b9e5" class="ne-p"><span class="ne-text">[19] </span><strong><span class="ne-text">Self-Supervised Monocular Depth and Ego-Motion Estimation in Endoscopy: Appearance Flow to the Rescue（Medical Image Analysis 2022）</span></strong><span class="ne-text">：提出基于外观流（appearance flow）的自监督学习框架，用于联合估计内窥镜视频中的深度与相机自运动，有效缓解了内窥镜场景中非朗伯反射和光照变化对自监督学习的干扰问题。</span></p><p id="u5aedda3e" class="ne-p"><span class="ne-text">[20] </span><strong><span class="ne-text">Stereo Dense Scene Reconstruction and Accurate Localization for Learning-Based Navigation of Laparoscope in Minimally Invasive Surgery（IEEE Transactions on Biomedical Engineering 2022）</span></strong><span class="ne-text">：基于双目腹腔镜系统，提出稠密三维重建与高精度位姿估计方法，为学习驱动的微创手术导航提供可靠的几何输入，相比单目方法在尺度恢复与重建精度方面具有明显优势。</span></p></details>
近年来，基于神经辐射场（Neural Radiance Fields，NeRF）的神经渲染技术为稠密手术场景重建提供了新视角合成能力 _[22]–[27]_。<u>神经隐式场表示进一步推动了 SLAM 系统的发展</u> _[28]–[33]_，<u>使其能够生成高质量的新视角图像和深度图</u>。例如，开创性工作 iMap 和 NICE-SLAM 分别采用多层感知机（MLP）和神经隐式网格进行场景表示。<u>然而，这类基于光线的体渲染方法计算开销巨大，难以在效率与精度之间取得平衡</u>。

<details class="lake-collapse"><summary id="u937ca341"><em><span class="ne-text">[22]–[27]</span></em></summary><p id="ud0aed32d" class="ne-p"><span class="ne-text">[22] </span><strong><span class="ne-text">Neural Rendering for Stereo 3D Reconstruction of Deformable Tissues in Robotic Surgery（MICCAI 2022）</span></strong><span class="ne-text">：将神经渲染思想引入机器人辅助手术中的双目三维重建问题，通过学习隐式场表示来建模可变形软组织的几何与外观变化，在复杂形变和非刚性运动条件下实现了比传统多视几何方法更稳定的重建效果，开启了内窥镜可变形组织神经重建的研究方向。</span></p><p id="uc638d81d" class="ne-p"><span class="ne-text">[23] </span><strong><span class="ne-text">Efficient Deformable Tissue Reconstruction via Orthogonal Neural Plane（arXiv 2023）</span></strong><span class="ne-text">：提出基于正交神经平面（Orthogonal Neural Plane）的高效隐式表示方法，在保持重建精度的同时显著降低了计算和存储开销，使可变形组织的神经重建更加高效，适用于对实时性要求较高的手术场景。</span></p><p id="u64dbf468" class="ne-p"><span class="ne-text">[24] </span><strong><span class="ne-text">Neural Lerplane Representations for Fast 4D Reconstruction of Deformable Tissues（arXiv 2023）</span></strong><span class="ne-text">：提出一种多平面神经表示（Lerplane），通过显式引入时间维度，实现对可变形组织随时间变化的四维（3D + 时间）重建，在速度与重建质量之间取得良好平衡，为动态软组织建模提供了有效的神经表示框架。</span></p><p id="u728371b7" class="ne-p"><span class="ne-text">[25] </span><strong><span class="ne-text">Endo-4DGS</span></strong><strong><span class="ne-text" style="color: #DF2A3F">*</span></strong><strong><span class="ne-text">: Endoscopic Monocular Scene Reconstruction with 4D Gaussian Splatting（arXiv 2024）</span></strong><span class="ne-text">：首次将四维高斯溅射（4D Gaussian Splatting）引入单目内窥镜场景重建，通过高斯分布同时建模空间几何与时间形变，实现了无需多视或双目系统的动态内窥镜场景重建，代表了 3DGS 向内窥镜可变形场景扩展的重要进展。</span></p><p id="uf342d1f1" class="ne-p"><span class="ne-text">[26] </span><strong><span class="ne-text">EndoGaussian</span></strong><strong><span class="ne-text" style="color: #DF2A3F">*</span></strong><strong><span class="ne-text">: Gaussian Splatting for Deformable Surgical Scene Reconstruction（2024）</span></strong><span class="ne-text">：提出基于高斯溅射的可变形手术场景重建方法，以显式点云高斯表示替代隐式神经场，在保证高质量几何与渲染效果的同时显著提升了优化与渲染效率，为实时或近实时手术导航与可视化提供了新的可行方案。</span></p><p id="u54678c73" class="ne-p"><span class="ne-text">[27] </span><strong><span class="ne-text">Deformable Endoscopic Tissues Reconstruction with Gaussian Splatting（arXiv 2024）</span></strong><strong><span class="ne-text" style="color: #DF2A3F">*</span></strong><span class="ne-text">：系统探索高斯溅射在内窥镜可变形组织重建中的应用，通过针对软组织形变特性的高斯参数化与优化策略，实现了稳定的几何重建与外观一致性，进一步验证了 3DGS 类方法在复杂医学内窥镜场景中的通用性与潜力。</span></p></details>
<details class="lake-collapse"><summary id="u74b0f7f2"><em><span class="ne-text">[28]–[33]</span></em></summary><p id="u47e5c49e" class="ne-p"><span class="ne-text">[28] </span><strong><span class="ne-text">Dense RGB SLAM with Neural Implicit Maps（arXiv 2023）</span></strong><span class="ne-text">：提出基于神经隐式地图的稠密 RGB SLAM 框架，通过连续隐式场表示同时建模场景几何与外观信息，在无需显式网格或点云结构的情况下实现高质量稠密建图与位姿估计，展示了神经隐式表示在高精度 RGB SLAM 中的潜力。</span></p><p id="u24d499f4" class="ne-p"><span class="ne-text">[29] </span><strong><span class="ne-text">iMAP: Implicit Mapping and Positioning in Real-Time（ICCV 2021）</span></strong><span class="ne-text">：首次实现基于隐式神经表示的实时 SLAM 系统，将连续函数形式的场景表示与相机位姿联合优化，打破了传统显式地图结构的限制，开创了“Neural SLAM”研究方向，对后续神经隐式 SLAM 方法产生了深远影响。</span></p><p id="u4d6c1c79" class="ne-p"><span class="ne-text">[30] </span><strong><span class="ne-text">Co-SLAM: Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM（CVPR 2023）</span></strong><span class="ne-text">：提出联合坐标编码与稀疏参数化编码的神经 SLAM 框架，在保证实时性能的同时显著提升了隐式场的表达效率与可扩展性，实现了在大规模场景中的稳定建图与定位。</span></p><p id="uf70aee71" class="ne-p"><span class="ne-text">[31] </span><strong><span class="ne-text">NICE-SLAM</span></strong><strong><span class="ne-text" style="color: #DF2A3F">*</span></strong><strong><span class="ne-text">: Neural Implicit Scalable Encoding for SLAM（CVPR 2022）</span></strong><span class="ne-text">：通过多分辨率层级隐式编码结构，将局部与全局几何信息解耦表示，有效提升了神经 SLAM 在大规模场景中的可扩展性和重建质量，是隐式神经表示与 SLAM 结合的重要代表性工作。</span></p><p id="ua8d655a1" class="ne-p"><span class="ne-text">[32] </span><strong><span class="ne-text">NICER-SLAM: Neural Implicit Scene Encoding for RGB SLAM（arXiv 2023）</span></strong><span class="ne-text">：在 NICE-SLAM 基础上进一步改进隐式表示与优化策略，增强了对 RGB 信息的建模能力，提高了在复杂光照与纹理条件下的鲁棒性，推动了纯 RGB 神经 SLAM 向更高精度与稳定性发展。</span></p><p id="u527542c8" class="ne-p"><span class="ne-text">[33] </span><strong><span class="ne-text">PointSLAM: Dense Neural Point Cloud-Based SLAM（ICCV 2023）</span></strong><span class="ne-text">：提出基于神经点云的显式—隐式混合表示 SLAM 方法，通过为每个点引入可学习的神经特征，实现稠密点云级别的建图与位姿估计，在效率与表达能力之间取得良好平衡，为后续基于 3D Gaussian / 点基神经表示的 SLAM 方法提供了重要启示。</span></p></details>
相比之下，<u>基于三维高斯溅射（3D Gaussian Splatting，3DGS） 的 SLAM 系统 </u>_<u>[35]–[37]</u>_<u> 在新视角渲染中同时实现了高速性与高保真度，为手术场景下的实时可视化与建图提供了新的可能性</u> _[38]_。<u>尽管具有上述优势，3DGS 方法无法渲染表面法向，并且由于多视角不一致性问题，难以实现精确的深度与表面细节重建</u> _[39]_。这一局限在手术场景中尤为突出，因为精确的空间表示至关重要。

<details class="lake-collapse"><summary id="u94d50bcc"><em><span class="ne-text">[35]–[37]</span></em></summary><p id="u7c6462be" class="ne-p"><span class="ne-text">[35] </span><strong><span class="ne-text">Gaussian Splatting SLAM（CVPR 2024）</span></strong><span class="ne-text">：首次系统性地将 3D Gaussian Splatting 引入 SLAM 框架，通过以高斯作为显式可微场景表示，实现相机位姿估计与高质量稠密建图的联合优化，在保持实时渲染能力的同时显著提升了地图的几何与外观表达能力，标志着 SLAM 从隐式神经场向高效显式神经表示的重要转变。</span></p><p id="u7ff45fe3" class="ne-p"><span class="ne-text">[36] </span><strong><span class="ne-text">GSSLAM: Dense Visual SLAM with 3D Gaussian Splatting（CVPR 2024）</span></strong><span class="ne-text">：提出基于 3D 高斯溅射的稠密视觉 SLAM 系统，将高斯参数优化与位姿估计紧密耦合，实现了高质量稠密重建与稳定定位，在复杂场景中相较传统点云或隐式表示具有更高的重建精度与优化效率。</span></p><p id="ue95e7035" class="ne-p"><span class="ne-text">[37] </span><strong><span class="ne-text">SplaTAM: Splat, Track &amp; Map 3D Gaussians for Dense RGB-D SLAM（CVPR 2024）</span></strong><span class="ne-text">：面向 RGB-D 场景提出基于 3D 高斯的跟踪与建图一体化框架，充分利用深度观测约束提升位姿估计精度与尺度稳定性，实现高效、鲁棒的稠密 SLAM，为 3D Gaussian Splatting 在多传感器 SLAM 中的应用提供了重要实践。</span></p></details>
<details class="lake-collapse"><summary id="u79b2fba0"><em><span class="ne-text">[38]</span></em></summary><p id="uf74d0401" class="ne-p"><span class="ne-text"> [38] </span><strong><span class="ne-text">EndoGSLAM</span></strong><strong><span class="ne-text" style="color: #DF2A3F">*</span></strong><strong><span class="ne-text">: Real-Time Dense Reconstruction and Tracking in Endoscopic Surgeries Using Gaussian Splatting（arXiv 2024）</span></strong><span class="ne-text">：提出面向内窥镜手术场景的实时 SLAM 框架，将 3D Gaussian Splatting 引入内窥镜位姿跟踪与稠密重建过程，通过高斯显式表示联合优化相机位姿与场景几何，在弱纹理、强反光和软组织形变等复杂条件下实现高质量重建与稳定跟踪，是 3DGS-SLAM 在内窥镜手术应用中的代表性工作。  </span></p></details>
<details class="lake-collapse"><summary id="u00039ec0"><em><span class="ne-text">[39]</span></em><span class="ne-text"> </span></summary><p id="u84a7dd08" class="ne-p"><span class="ne-text">[39] </span><strong><span class="ne-text">2D Gaussian Splatting for Geometrically Accurate Radiance Fields（SIGGRAPH 2024）</span></strong><span class="ne-text">：提出以二维高斯为基本表示的可微渲染框架，通过在像平面中进行高斯溅射并引入严格的几何一致性约束，实现对辐射场的高精度建模，在几何准确性与渲染效率之间取得良好平衡，为高斯溅射表示在高精度几何重建与实时渲染中的进一步发展提供了新的理论与方法基础。  </span></p></details>
<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767598652409-0baf9fa0-6287-4554-8483-8812b29c5d45.png)

**图 1：重建与渲染结果示意**  
与基于 3DGS 的 SLAM 方法相比，本文方法采用 2DGS 进行几何精确的场景表示，能够生成高质量的新视角渲染图像、视角一致的深度图以及精确的表面法向信息。  

为解决上述问题，本文提出了一种基于二维高斯溅射（2D Gaussian Splatting，2DGS）的创新型实时跟踪与建图系统——**Endo-2DTAM**，用于内窥镜场景重建，如_图 1 _所示。所提出的系统由三个模块组成：**跟踪模块、建图模块以及束调整模块**。<u>得益于 2DGS 对表面法向的显式表示能力，我们的跟踪模块同时利用点到点（point-to-point）和点到平面（point-to-plane）距离度量，从而在位姿估计过程中引入表面法向信息</u>。<u>随后，建图模块联合考虑法向一致性与深度畸变，以提升表面重建质量</u>。<u>此外，本文提出了一种</u>**<u>位姿一致的关键帧采样策略</u>**<u>，通过在建图与束调整阶段对所选关键帧进行优化，在保证覆盖完整性的同时兼顾计算效率</u>。

据我们所知，<u>本文首次将 2DGS 引入 SLAM 系统中，用以解决以往基于 3DGS 的 SLAM 方法 </u>_<u>[38]</u>_<u> 中长期存在的多视角不一致问题</u>。我们在公开数据集上对所提出的方法进行了评估，并在几何重建任务中取得了当前最优性能，验证了其在手术应用中的潜力。

<details class="lake-collapse"><summary id="ufe6a59d7"><em><span class="ne-text">[38]</span></em></summary><p id="u15872fa3" class="ne-p"><span class="ne-text"> [38] </span><strong><span class="ne-text">EndoGSLAM</span></strong><strong><span class="ne-text" style="color: #DF2A3F">*</span></strong><strong><span class="ne-text">: Real-Time Dense Reconstruction and Tracking in Endoscopic Surgeries Using Gaussian Splatting（arXiv 2024）</span></strong><span class="ne-text">：提出面向内窥镜手术场景的实时 SLAM 框架，将 3D Gaussian Splatting 引入内窥镜位姿跟踪与稠密重建过程，通过高斯显式表示联合优化相机位姿与场景几何，在弱纹理、强反光和软组织形变等复杂条件下实现高质量重建与稳定跟踪，是 3DGS-SLAM 在内窥镜手术应用中的代表性工作。  </span></p></details>
**本文的主要贡献总结如下****<font style="color:#DF2A3F;">*</font>****：**

• 提出了一种基于二维高斯溅射的全新 SLAM 系统，实现了内窥镜场景中精确的相机跟踪与高保真的三维组织重建。  
• 系统能够实现实时新视角渲染，生成逼真的 RGB 图像、视角一致的深度结果以及精确的表面法向。  
• 提出了一种面向表面法向的跟踪与建图流程，并结合位姿一致的关键帧策略，实现高精度几何重建。  
• 在公开数据集上进行了大量实验，在内窥镜场景的深度重建任务中取得了仅为 1.87 ± 0.63 mm 的 RMSE，达到了当前最优水平。

## RELATED WORK
### 基于神经场的 SLAM（Neural Field based SLAM）
近年来，神经场方法（NeRF 系列）在稠密重建与高质量新视角合成方面取得了显著进展 _[21]、[40]、[41]_，并已被进一步拓展至内窥镜场景重建任务中 _[22]、[23]、[24]_。其中，基于_隐式 _表示的方法，如 iMAP、MeSLAM _[42]_ 和 Go-SLAM _[43]_，采用多层感知机（MLP）进行场景表示，并取得了较为理想的效果。相比之下，基于_显式 _表示的方法，如 ESLAM _[44]_、Point-SLAM _[33]_ 和 Loopy-SLAM _[45]_，引入了三平面（tri-plane）或神经点云表示，从而提升了建图精度与重建保真度。

<details class="lake-collapse"><summary id="u9ca94b0f"><em><span class="ne-text">隐式/显式</span></em><span class="ne-text">：</span><span class="ne-text" style="text-decoration: underline">隐式方法</span><span class="ne-text">通过连续神经函数对场景进行整体建模，具有表示紧凑、连续性强的优势，但计算代价较高（如 NeRF ）；</span><span class="ne-text" style="text-decoration: underline">显式方法</span><span class="ne-text">则通过可解析的几何代理（如点、网格或高斯）对场景进行直接存储，在效率与可扩展性方面更具优势。  </span></summary><p id="uf137b454" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/45861457/1767686160412-325ed468-9726-4346-a0b2-534e65c4f546.png" width="497.7478832037186" title="" crop="0,0,1,1" id="u1c92f774" class="ne-image"></p><p id="u01c598e9" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/45861457/1767686177957-ffda48ca-da26-4977-a88c-bb693e45f08c.png" width="475.15964863369567" title="" crop="0,0,1,1" id="u38c0998e" class="ne-image"></p></details>
<details class="lake-collapse"><summary id="u977487c7"><em><span class="ne-text"> [21]、[40]、[41]</span></em></summary><p id="u22250287" class="ne-p"><span class="ne-text">[21] </span><strong><span class="ne-text">NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis（Communications of the ACM 2021）</span></strong><span class="ne-text">：提出神经辐射场（Neural Radiance Fields, NeRF）表示，将三维空间中的连续体素位置与视角映射为颜色和体密度函数，实现高质量新视角合成，奠定了以隐式神经表示建模三维场景几何与外观的基础性范式，对后续神经渲染、神经重建及神经 SLAM 研究产生了深远影响。</span></p><p id="u004dd73c" class="ne-p"><span class="ne-text">[40] </span><strong><span class="ne-text">NeRF++: Analyzing and Improving Neural Radiance Fields（arXiv 2020）</span></strong><span class="ne-text">：针对原始 NeRF 难以建模无界场景的问题，引入前景—背景分离建模策略，对 NeRF 表示能力与局限性进行了系统分析并加以改进，使神经辐射场能够适用于更大尺度和更复杂的真实场景。</span></p><p id="ud976e69c" class="ne-p"><span class="ne-text">[41] </span><strong><span class="ne-text">NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections（CVPR 2021）</span></strong><span class="ne-text">：扩展 NeRF 至非受控互联网照片集合场景，通过联合建模相机曝光、光照变化和颜色偏差，实现对真实复杂环境下多视图数据的鲁棒建模，显著提升了 NeRF 在实际应用场景中的适用性。</span></p></details>
<details class="lake-collapse"><summary id="u056797d2"><em><span class="ne-text">[22]、[23]、[24]</span></em></summary><p id="u1c04f42c" class="ne-p"><span class="ne-text">[22] </span><strong><span class="ne-text">Neural Rendering for Stereo 3D Reconstruction of Deformable Tissues in Robotic Surgery（MICCAI 2022）</span></strong><span class="ne-text">：将神经渲染思想引入机器人辅助手术中的双目三维重建问题，通过学习隐式场表示来建模可变形软组织的几何与外观变化，在复杂形变和非刚性运动条件下实现了比传统多视几何方法更稳定的重建效果，开启了内窥镜可变形组织神经重建的研究方向。</span></p><p id="u59c6f998" class="ne-p"><span class="ne-text">[23] </span><strong><span class="ne-text">Efficient Deformable Tissue Reconstruction via Orthogonal Neural Plane（arXiv 2023）</span></strong><span class="ne-text">：提出基于正交神经平面（Orthogonal Neural Plane）的高效隐式表示方法，在保持重建精度的同时显著降低了计算和存储开销，使可变形组织的神经重建更加高效，适用于对实时性要求较高的手术场景。</span></p><p id="u9b02f8f5" class="ne-p"><span class="ne-text">[24] </span><strong><span class="ne-text">Neural Lerplane Representations for Fast 4D Reconstruction of Deformable Tissues（arXiv 2023）</span></strong><span class="ne-text">：提出一种多平面神经表示（Lerplane），通过显式引入时间维度，实现对可变形组织随时间变化的四维（3D + 时间）重建，在速度与重建质量之间取得良好平衡，为动态软组织建模提供了有效的神经表示框架。</span></p></details>
<details class="lake-collapse"><summary id="uf6b4f8dc"><em><span class="ne-text">[42]  [43]</span></em></summary><p id="u16db7b5a" class="ne-p"><span class="ne-text">[42] </span><strong><span class="ne-text">MESLAM: Memory Efficient SLAM Based on Neural Fields（IEEE SMC 2022）</span></strong><span class="ne-text">：提出一种以内存效率为核心设计目标的神经场 SLAM 框架，通过紧凑的神经隐式表示与增量式优化策略，在显著降低内存开销的同时保持较好的定位与建图性能，为神经场 SLAM 在资源受限平台上的实际部署提供了可行方案。</span></p><p id="uc8dc6abb" class="ne-p"><span class="ne-text">[43] </span><strong><span class="ne-text">GO-SLAM: Global Optimization for Consistent 3D Instant Reconstruction（ICCV 2023）</span></strong><span class="ne-text">：提出面向即时三维重建的全局一致性优化 SLAM 框架，通过引入全局约束与跨帧一致性优化，有效缓解增量式重建中误差累积与漂移问题，实现更稳定、更一致的三维结构重建，体现了从局部增量 SLAM 向全局优化神经 SLAM 的发展趋势。</span></p></details>
<details class="lake-collapse"><summary id="ubf1deaad"><em><span class="ne-text">[44] [33] [45]</span></em></summary><p id="u855d5049" class="ne-p"><span class="ne-text">[44] </span><strong><span class="ne-text">ESLAM: Efficient Dense SLAM System Based on Hybrid Representation of Signed Distance Fields（CVPR 2023）</span></strong><span class="ne-text">：提出一种基于混合表示的高效稠密 SLAM 系统，将符号距离场（SDF）的几何优势与紧凑的数据结构相结合，在保证重建精度的同时显著提升了计算效率与内存利用率，为稠密 SLAM 在实时应用中的落地提供了工程化可行路径。</span></p><p id="u193ac4ce" class="ne-p"><span class="ne-text">[33] </span><strong><span class="ne-text">PointSLAM: Dense Neural Point Cloud-Based SLAM（ICCV 2023）</span></strong><span class="ne-text">：提出基于神经点云的稠密 SLAM 框架，为每个点引入可学习的神经特征，实现显式点云表示与隐式神经建模的融合，在效率、可扩展性与表达能力之间取得良好平衡，为后续基于点/高斯的神经 SLAM 方法（如 3DGS-SLAM）提供了重要启发。</span></p><p id="uc0422716" class="ne-p"><span class="ne-text">[45] </span><strong><span class="ne-text">Loopy-SLAM: Dense Neural SLAM with Loop Closures（CVPR 2024）</span></strong><span class="ne-text">：针对早期神经 SLAM 方法缺乏回环闭合机制的问题，引入基于神经表示的回环检测与全局一致性优化策略，在保持稠密重建质量的同时有效抑制长期漂移，显著提升了大规模场景中的全局一致性与鲁棒性，标志着神经 SLAM 向完整系统化方向迈进。</span></p></details>
此外，混合式方法（hybrid-based），例如 NICE-SLAM _[31]_、NGEL-SLAM _[46]_ 和 CoSLAM _[30]_，采用神经网格（neural grid）表示，并通过联合基于坐标的特征与参数化编码来优化渲染过程。<u>然而，由于</u>_<u>体渲染 </u>_<u>本身具有较高的计算开销，基于神经场的 SLAM 系统在精度与效率之间仍难以取得良好的平衡</u>。

<details class="lake-collapse"><summary id="ue0d6c836"><em><span class="ne-text">体渲染</span></em><span class="ne-text"> ：NeRF 的渲染方式， 即通过</span><strong><span class="ne-text">沿视线对连续体数据进行积分</span></strong><span class="ne-text">来生成图像的渲染方式。  </span></summary><p id="u0c2a0085" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/45861457/1768113555312-0ee81217-8196-4bbf-bec6-f8217d30f36b.png" width="800.2688819093822" title="" crop="0,0,1,1" id="u28172833" class="ne-image"></p></details>
<details class="lake-collapse"><summary id="ub3d17a6c"><em><span class="ne-text">[31] [46] [30]</span></em></summary><p id="ub1ed542e" class="ne-p"><span class="ne-text">[31] </span><strong><span class="ne-text">NICE-SLAM</span></strong><strong><span class="ne-text" style="color: #DF2A3F">*</span></strong><strong><span class="ne-text">: Neural Implicit Scalable Encoding for SLAM（CVPR 2022）</span></strong><span class="ne-text">：通过多分辨率层级隐式编码结构，将局部与全局几何信息解耦表示，有效提升了神经 SLAM 在大规模场景中的可扩展性和重建质量，是隐式神经表示与 SLAM 结合的重要代表性工作。</span></p><p id="u140aa5e6" class="ne-p"><span class="ne-text">[46] </span><strong><span class="ne-text">NGEL-SLAM: Neural Implicit Representation-Based Global Consistent Low-Latency SLAM System（ICRA 2024）</span></strong><span class="ne-text">：提出一种基于神经隐式表示的低时延 SLAM 系统，通过引入全局一致性约束与高效的增量式优化策略，在保持实时性能的同时有效抑制长期漂移，实现稳定的全局一致定位与建图，体现了神经隐式 SLAM 向工程可用性与系统完整性发展的趋势。  </span></p><p id="u3c63c98b" class="ne-p"><span class="ne-text">[30] </span><strong><span class="ne-text">Co-SLAM: Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM（CVPR 2023）</span></strong><span class="ne-text">：提出联合坐标编码与稀疏参数化编码的神经 SLAM 框架，在保证实时性能的同时显著提升了隐式场的表达效率与可扩展性，实现了在大规模场景中的稳定建图与定位。</span></p></details>
### **高斯溅射（Gaussian Splatting）**
近年来，基于三维高斯溅射（3D Gaussian Splatting，3DGS）的神经渲染方法因其在实时新视角合成任务中表现出卓越性能而受到广泛关注。后续研究工作 _[47]–[50]_ 进一步引入多层感知机（MLP）、四维基元（4D primitives）以及 HexPlane 等表示方式来建模时间信息，从而实现了对动态场景的重建。<u>得益于其出色的渲染效率与效果，3DGS 这一基于显式场景表示的技术迅速被应用于手术重建任务中 </u>_<u>[25]、[26]、[27]</u>_<u>，实现了高质量的可变形组织重建</u>。

<u>然而，3DGS 并未对表面法向进行明确建模，导致重建结果在不同视角下存在几何不一致性问题。为解决这一局限，二维高斯溅射（2D Gaussian Splatting，2DGS）</u>_<u>[39]</u>_<u> 被提出，并提供了具有明确定义的表面法向表示</u>。**基于此，本文提出了一种以 2DGS 作为场景表示的全新 SLAM 系统，相较于基于 3DGS 的 SLAM 方法，能够实现更加视角一致的几何重建**。

<details class="lake-collapse"><summary id="u585cd3e2"><em><span class="ne-text">[47]–[50]</span></em></summary><p id="u61ebb2b1" class="ne-p"><span class="ne-text">[47] </span><strong><span class="ne-text">Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis（3DV 2024）</span></strong><span class="ne-text">：提出面向动态场景的 3D 高斯表示方法，通过引入具有时间一致性的可持久高斯（persistent Gaussians），在视角合成过程中实现对动态物体的持续跟踪与建模，为高斯溅射在非静态场景中的应用提供了新的建模范式。</span></p><p id="u31874d2b" class="ne-p"><span class="ne-text">[48] </span><strong><span class="ne-text">4D Gaussian Splatting for Real-Time Dynamic Scene Rendering（CVPR 2024）</span></strong><span class="ne-text">：提出基于四维高斯（空间 + 时间）的动态场景表示与渲染框架，将时间维度显式纳入高斯参数化过程，在保证高渲染质量的同时实现实时动态场景建模，显著提升了高斯溅射在动态环境中的表达能力。</span></p><p id="u5061a704" class="ne-p"><span class="ne-text">[49] </span><strong><span class="ne-text">Real-Time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting（ICLR 2024）</span></strong><span class="ne-text">：进一步完善 4D Gaussian Splatting 的动态建模能力，通过高效的参数更新与渲染策略，实现对复杂动态场景的高保真、实时表示与渲染，验证了 4D 高斯表示在动态场景神经渲染中的通用性与实用性。</span></p><p id="uf73587da" class="ne-p"><span class="ne-text">[50] </span><strong><span class="ne-text">Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction（arXiv 2023）</span></strong><span class="ne-text">：提出可变形 3D 高斯表示，通过对高斯参数引入形变建模与时序约束，实现单目条件下的高质量动态场景重建，为高斯溅射方法在单目动态与可变形场景中的应用奠定了基础。</span></p></details>
<details class="lake-collapse"><summary id="uecb510ea"><em><span class="ne-text">[25]、[26]、[27]</span></em><em><span class="ne-text" style="color: #DF2A3F">*</span></em></summary><p id="u847994ac" class="ne-p"><span class="ne-text">[25] </span><strong><span class="ne-text">Endo-4DGS</span></strong><strong><span class="ne-text" style="color: #DF2A3F">*</span></strong><strong><span class="ne-text">: Endoscopic Monocular Scene Reconstruction with 4D Gaussian Splatting（arXiv 2024）</span></strong><span class="ne-text">：首次将四维高斯溅射（4D Gaussian Splatting）引入单目内窥镜场景重建，通过高斯分布同时建模空间几何与时间形变，实现了无需多视或双目系统的动态内窥镜场景重建，代表了 3DGS 向内窥镜可变形场景扩展的重要进展。</span></p><p id="uc1c5108b" class="ne-p"><span class="ne-text">[26] </span><strong><span class="ne-text">EndoGaussian</span></strong><strong><span class="ne-text" style="color: #DF2A3F">*</span></strong><strong><span class="ne-text">: Gaussian Splatting for Deformable Surgical Scene Reconstruction（2024）</span></strong><span class="ne-text">：提出基于高斯溅射的可变形手术场景重建方法，以显式点云高斯表示替代隐式神经场，在保证高质量几何与渲染效果的同时显著提升了优化与渲染效率，为实时或近实时手术导航与可视化提供了新的可行方案。</span></p><p id="u1caa4bda" class="ne-p"><span class="ne-text">[27] </span><strong><span class="ne-text">Deformable Endoscopic Tissues Reconstruction with Gaussian Splatting（arXiv 2024）</span></strong><strong><span class="ne-text" style="color: #DF2A3F">*</span></strong><span class="ne-text">：系统探索高斯溅射在内窥镜可变形组织重建中的应用，通过针对软组织形变特性的高斯参数化与优化策略，实现了稳定的几何重建与外观一致性，进一步验证了 3DGS 类方法在复杂医学内窥镜场景中的通用性与潜力。</span></p></details>
<details class="lake-collapse"><summary id="u0cc48580"><em><span class="ne-text">[39]</span></em><em><span class="ne-text" style="color: #DF2A3F"></span></em></summary><p id="u7e9d01f1" class="ne-p"><span class="ne-text">[39] </span><strong><span class="ne-text">2D Gaussian Splatting for Geometrically Accurate Radiance Fields（SIGGRAPH 2024）</span></strong><strong><span class="ne-text" style="color: #DF2A3F">*</span></strong><span class="ne-text">：提出以二维高斯为基本表示的可微渲染框架，通过在像平面中进行高斯溅射并引入严格的几何一致性约束，实现对辐射场的高精度建模，在几何准确性与渲染效率之间取得良好平衡，为高斯溅射表示在高精度几何重建与实时渲染中的进一步发展提供了新的理论与方法基础。  </span></p></details>
### **内窥镜 SLAM（Endoscopic SLAM）**
内窥镜 SLAM 由于手术环境受限、视场范围狭窄、纹理信息稀缺以及组织形变等因素，面临着一系列_独特挑战_ _[4]_。<u>早期研究</u>工作 _[8]、[9]、[51]_ 主要<u>采用基于特征的跟踪方法，以应对复杂多变的光照条件</u>。<u>针对纹理稀缺问题</u>，一些研究 _[15]、[52]、[53]、[54]_ <u>提出了诸如标记点跟踪（marker tracking）和特征对应分析等解决方案</u>。

<details class="lake-collapse"><summary id="u66005d7f"><em><span class="ne-text">独特挑战 [4]</span></em><span class="ne-text">： 弱纹理、强反光、非刚性形变等挑战。</span></summary><p id="u50accc6c" class="ne-p"><span class="ne-text">[4] </span><strong><span class="ne-text">The Future of Endoscopic Navigation: A Review of Advanced Endoscopic Vision Technology（IEEE Access 2021）</span></strong><span class="ne-text">：系统综述了内窥镜导航领域的前沿视觉技术发展，涵盖内窥镜位姿跟踪、三维重建、视觉 SLAM、深度估计以及学习驱动方法，重点分析了弱纹理、强反光、非刚性形变等内窥镜典型挑战，总结了现有方法的优势与局限，并对智能化内窥镜导航系统的未来研究方向进行了展望。  </span></p></details>
<details class="lake-collapse"><summary id="uce471996"><em><span class="ne-text">[8]、[9]、[51]</span></em></summary><p id="ub51702ae" class="ne-p"><span class="ne-text">[8] </span><strong><span class="ne-text">Visual SLAM for Handheld Monocular Endoscope（IEEE Transactions on Medical Imaging 2013）</span></strong><span class="ne-text">：首次系统性地将单目视觉 SLAM 引入手持式内窥镜场景，通过基于特征点的位姿估计与稀疏地图构建，实现内窥镜运动轨迹恢复与三维结构重建，验证了单目 SLAM 在内窥镜导航中的可行性，为后续相关研究奠定了基础。</span></p><p id="u2f2a61b4" class="ne-p"><span class="ne-text">[9] </span><strong><span class="ne-text">SLAM-Based Quasi-Dense Reconstruction for Minimally Invasive Surgery Scenes（arXiv 2017）</span></strong><span class="ne-text">：在传统稀疏特征 SLAM 框架基础上，引入准稠密（quasi-dense）重建策略，将多视几何与深度线索相结合，实现对微创手术场景更高覆盖率的三维重建，在保持实时性的同时显著提升了重建完整性。</span></p><p id="ud6bc4ded" class="ne-p"><span class="ne-text">[51] </span><strong><span class="ne-text">Live Tracking and Dense Reconstruction for Handheld Monocular Endoscopy（IEEE Transactions on Medical Imaging 2018）</span></strong><span class="ne-text">：进一步扩展前期工作，在单目内窥镜 SLAM 框架中实现实时位姿跟踪与稠密三维重建，通过改进的深度估计与融合策略显著提升了几何细节恢复能力，标志着单目内窥镜 SLAM 从稀疏/准稠密向稠密重建的重要进展。</span></p></details>
<details class="lake-collapse"><summary id="ub97cae72"><em><span class="ne-text">[15]、[52]、[53]、[54]</span></em></summary><p id="u9d1f5652" class="ne-p"><span class="ne-text">[15] </span><strong><span class="ne-text">RNNSLAM: Reconstructing the 3D Colon to Visualize Missing Regions during a Colonoscopy（Medical Image Analysis 2021）</span></strong><span class="ne-text">：利用循环神经网络建模结肠镜视频中的长期时序依赖关系，实现对不可见或遗漏区域的三维结构预测与补全，有效缓解了结肠镜检查中视野受限导致的结构缺失问题，为基于学习的内窥镜三维重建提供了新的时间建模思路。</span></p><p id="u65a76be0" class="ne-p"><span class="ne-text">[52] </span><strong><span class="ne-text">Endoscope Navigation and 3D Reconstruction of Oral Cavity by Visual SLAM with Mitigated Data Scarcity（CVPR Workshops 2018）</span></strong><span class="ne-text">：针对口腔内窥镜场景中真实数据稀缺的问题，提出结合合成数据与真实数据的视觉 SLAM 导航与三维重建方法，在有限标注条件下提升系统的鲁棒性与泛化能力，为内窥镜 SLAM 在受限数据环境下的应用提供了可行方案。</span></p><p id="u2505532a" class="ne-p"><span class="ne-text">[53] </span><strong><span class="ne-text">Unsupervised Odometry and Depth Learning for Endoscopic Capsule Robots（IROS 2018）</span></strong><span class="ne-text">：提出一种无监督学习框架，同时估计内窥镜胶囊机器人的相机位姿与深度信息，摆脱对真实深度或位姿标注的依赖，为端到端学习方法在内窥镜视觉里程计与三维感知中的应用奠定了基础。</span></p><p id="uf39f38ba" class="ne-p"><span class="ne-text">[54] </span><strong><span class="ne-text">Extremely Dense Point Correspondences Using a Learned Feature Descriptor（CVPR 2020）</span></strong><span class="ne-text">：提出基于深度学习的稠密特征描述子，实现图像间极高密度的点对应关系匹配，为弱纹理、强形变医学影像场景下的稠密配准与三维重建提供了关键技术支撑，广泛应用于内窥镜 SLAM、配准与形变建模任务中。</span></p></details>
<u>为提升稠密重建质量</u>，部分方法 _[13]–[20]_ <u>引入深度神经网络进行稠密估计与跟踪</u>。<u>针对非刚性组织运动</u>，研究工作中还<u>引入了基于图结构的特征表示方法 </u>_<u>[55]</u>_<u> 用于组织跟踪</u>。近期，<font style="background-color:#FCE75A;">EndoGSLAM</font> _[38]_ 通过实时新视角合成推动了内窥镜场景的三维重建，其高效的渲染性能使外科医生能够交互式地观察和检查三维场景中的任意区域。

<details class="lake-collapse"><summary id="u6df9968b"><em><span class="ne-text">[13]-[20]</span></em></summary><p id="u363c6814" class="ne-p"><span class="ne-text">[13] </span><strong><span class="ne-text">Vision–Kinematics Interaction for Robotic-Assisted Bronchoscopy Navigation（IEEE Transactions on Medical Imaging 2022）</span></strong><span class="ne-text">：提出视觉信息与机器人运动学模型深度融合的支气管镜导航框架，通过联合优化视觉观测与机械臂先验运动约束，显著提升了复杂支气管结构中位姿估计的稳定性与精度，证明了多源先验约束在机器人辅助手术导航中的重要价值。</span></p><p id="ueaccb8c3" class="ne-p"><span class="ne-text">[14] </span><strong><span class="ne-text">SAGE: SLAM with Appearance and Geometry Prior for Endoscopy（ICRA 2022）</span></strong><span class="ne-text">：提出一种结合外观先验与几何先验的端到端 SLAM 框架，将学习得到的场景外观表征与几何一致性约束引入传统 SLAM 优化过程，在弱纹理、强反光的内窥镜场景中显著提高了定位与重建的鲁棒性，代表了“学习辅助 SLAM”的典型范式。</span></p><p id="u6a49a8d6" class="ne-p"><span class="ne-text">[15] </span><strong><span class="ne-text">RNNSLAM: Reconstructing the 3D Colon to Visualize Missing Regions during a Colonoscopy（Medical Image Analysis 2021）</span></strong><span class="ne-text">：利用循环神经网络建模结肠镜视频中的长期时序依赖关系，实现对不可见或遗漏区域的三维结构预测与补全，有效缓解了结肠镜检查中视野受限导致的结构缺失问题，为基于学习的内窥镜三维重建提供了新的时间建模思路。</span></p><p id="u3b98f42c" class="ne-p"><span class="ne-text">[16] </span><strong><span class="ne-text">C³ Fusion: Consistent Contrastive Colon Fusion, Towards Deep SLAM in Colonoscopy（Shape in Medical Imaging Workshop 2023）</span></strong><span class="ne-text">：提出基于对比学习的一致性特征融合策略，将深度特征匹配与时序一致性约束相结合，实现结肠镜视频中的稳健位姿估计与地图融合，推动了从传统几何 SLAM 向深度学习驱动 SLAM 的过渡。</span></p><p id="u0435672b" class="ne-p"><span class="ne-text">[17] </span><strong><span class="ne-text">Bimodal Camera Pose Prediction for Endoscopy（IEEE Transactions on Medical Robotics and Bionics 2023）</span></strong><span class="ne-text">：针对内窥镜运动中存在的多解性问题，提出双模态相机位姿预测模型，通过同时建模不同可能运动假设，提高了位姿估计在快速运动和视角剧烈变化条件下的鲁棒性。</span></p><p id="u2f93576d" class="ne-p"><span class="ne-text">[18] </span><strong><span class="ne-text">EndoDepth-and-Motion: Reconstruction and Tracking in Endoscopic Videos Using Depth Networks and Photometric Constraints（RA-L 2021）</span></strong><span class="ne-text">：将单目深度学习网络与经典光度一致性约束相结合，实现端到端的内窥镜位姿跟踪与三维重建，在无需真实深度标注的情况下获得较为稳定的深度与运动估计，体现了“深度学习 + 几何约束”的混合建模思想。</span></p><p id="u9eb70218" class="ne-p"><span class="ne-text">[19] </span><strong><span class="ne-text">Self-Supervised Monocular Depth and Ego-Motion Estimation in Endoscopy: Appearance Flow to the Rescue（Medical Image Analysis 2022）</span></strong><span class="ne-text">：提出基于外观流（appearance flow）的自监督学习框架，用于联合估计内窥镜视频中的深度与相机自运动，有效缓解了内窥镜场景中非朗伯反射和光照变化对自监督学习的干扰问题。</span></p><p id="u3f603f7b" class="ne-p"><span class="ne-text">[20] </span><strong><span class="ne-text">Stereo Dense Scene Reconstruction and Accurate Localization for Learning-Based Navigation of Laparoscope in Minimally Invasive Surgery（IEEE Transactions on Biomedical Engineering 2022）</span></strong><span class="ne-text">：基于双目腹腔镜系统，提出稠密三维重建与高精度位姿估计方法，为学习驱动的微创手术导航提供可靠的几何输入，相比单目方法在尺度恢复与重建精度方面具有明显优势。</span></p></details>
<details class="lake-collapse"><summary id="u6b48b2a9"><em><span class="ne-text">[55]</span></em><em><span class="ne-text" style="color: #DF2A3F"></span></em></summary><p id="ua7e8cc1b" class="ne-p"><span class="ne-text">[55] </span><strong><span class="ne-text">NR-SLAM</span></strong><strong><span class="ne-text" style="color: #DF2A3F"></span></strong><strong><span class="ne-text">: Nonrigid Monocular SLAM（IEEE Transactions on Robotics, 2024）</span></strong><span class="ne-text">：针对真实场景中普遍存在的</span><strong><span class="ne-text">非刚性形变问题</span></strong><span class="ne-text">，提出了一种单目非刚性 SLAM 框架，在传统相机位姿估计的基础上，引入对场景形变的联合建模与优化。该方法通过对非刚性运动进行显式参数化，在保证相机跟踪稳定性的同时，实现了对动态、可变形场景的几何重建，为内窥镜等存在软组织形变的应用场景提供了重要参考。  </span></p></details>
<details class="lake-collapse"><summary id="u1200bdc8"><em><span class="ne-text">[38]</span></em><em><span class="ne-text" style="color: #DF2A3F">*</span></em></summary><p id="u3dd099d2" class="ne-p"><span class="ne-text">[38] </span><strong><span class="ne-text">EndoGSLAM</span></strong><strong><span class="ne-text" style="color: #DF2A3F">*</span></strong><strong><span class="ne-text">: Real-Time Dense Reconstruction and Tracking in Endoscopic Surgeries Using Gaussian Splatting（arXiv 2024）</span></strong><span class="ne-text">：提出面向内窥镜手术场景的实时 SLAM 框架，将 3D Gaussian Splatting 引入内窥镜位姿跟踪与稠密重建过程，通过高斯显式表示联合优化相机位姿与场景几何，在弱纹理、强反光和软组织形变等复杂条件下实现高质量重建与稳定跟踪，是 3DGS-SLAM 在内窥镜手术应用中的代表性工作。  </span></p></details>
<u>尽管取得了上述进展，内窥镜 SLAM 在</u>**<u>重建质量</u>**<u>和</u>**<u>时间效率</u>**<u>方面仍然面临显著挑战</u>。为此，本文提出了一种新的内窥镜 SLAM 系统，在保证实时性的同时，实现了当前最优（state-of-the-art，SOTA）的几何重建精度。

## METHODOLOGY
<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767598744047-a2e287f4-2b08-472e-9e35-76796f704748.png)

**图 2. Endo-2DTAM 系统总体框架。**  
本文提出的系统由三个模块组成：跟踪模块、建图模块以及束调整模块。跟踪模块以输入的 RGB-D 帧为输入，完成相机位姿估计；随后，该帧被加入候选列表，用于后续的位姿一致关键帧选择。在建图模块中，首先利用新输入帧对二维高斯进行扩展，然后基于选定的关键帧对二维高斯进行更新。所选关键帧同时用于束调整过程，以实现相机位姿与二维高斯参数的联合优化。

Endo-2DTAM 是一种基于二维高斯溅射（2D Gaussian Splatting）的稠密 RGB-D SLAM 系统。_图 2_ 展示了本文所提出系统的整体框架。在本节中，我们将从以下几个方面对系统的具体细节进行说明：二维高斯表示（[1.3.1](#NXT4Q)）、跟踪模块（[1.3.2](#naV1M)）、高斯扩展与关键帧采样（[1.3.3](#Gnm5l)）、建图模块（[1.3.4](#tTKbL)）以及束调整（[1.3.5](#cjVJO)）。

###  二维高斯场景表示（2D Gaussian Scene Representation）
为实现几何精确的内窥镜场景表示并支持实时高质量渲染，我们采用一组二维高斯对内窥镜场景进行建模：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767599133488-d2c33764-7373-4166-900a-b2eefeb5a938.png)

其中，每个二维高斯$ G_i $由位置$ \mathbf{X}_i\in\mathbb{R}^3 $、旋转矩阵$ \mathbf{R}_i\in\mathbb{R}^{3\times3} $、尺度矩阵$ \mathbf{S}_i\in\mathbb{R}^{3\times3} $、不透明度$ \Lambda_i\in\mathbb{R} $以及 RGB 颜色$ \mathbf{C}_i\in\mathbb{R}^3 $ 所定义。旋转矩阵表示为$ \mathbf{R}_i=[\mathbf{t}_u,\mathbf{t}_v,\mathbf{t}_w] $，其中$ \mathbf{t}_u,\mathbf{t}_v $为两条相互正交的切向量，$ \mathbf{t}_w=\mathbf{t}_u\times\mathbf{t}_v $ 表示该高斯基元的法向量。按照文献 _[39]_ 的定义，尺度矩阵$ \mathbf{S}_i $为对角矩阵，其最后一个对角元素为 0，其余两个元素为 $ (s_u,s_v) $。

<details class="lake-collapse"><summary id="u80bbf817"><em><span class="ne-text">[39]</span></em><em><span class="ne-text" style="color: #DF2A3F"></span></em></summary><p id="u938dd567" class="ne-p"><span class="ne-text">[39] </span><strong><span class="ne-text">2D Gaussian Splatting for Geometrically Accurate Radiance Fields（SIGGRAPH 2024）</span></strong><strong><span class="ne-text" style="color: #DF2A3F">*</span></strong><span class="ne-text">：提出以二维高斯为基本表示的可微渲染框架，通过在像平面中进行高斯溅射并引入严格的几何一致性约束，实现对辐射场的高精度建模，在几何准确性与渲染效率之间取得良好平衡，为高斯溅射表示在高精度几何重建与实时渲染中的进一步发展提供了新的理论与方法基础。  </span></p></details>
```python
# utils\slam_helpers.py L438-L449
def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, use_simplification=False):
    """
    初始化 Gaussian SLAM 的全局参数和变量。
    输入:
        init_pt_cld: 初始点云 (N, 6)，包含 XYZ 和 RGB
        num_frames: 总帧数，用于初始化相机轨迹
        mean3_sq_dist: 点与点之间的平均平方距离
        use_simplification: 是否简化模型（如不使用球谐函数）
    作用:
        创建 Gaussian 属性（位置、颜色、旋转、不透明度、缩放）和相机轨迹（旋转、平移）的参数张量。
    输出:
        params: 包含所有可优化参数的字典
        variables: 包含辅助状态（如梯度、时间戳、半径）的字典
    """
    num_pts = init_pt_cld.shape[0]  # 点的数量
    means3D = init_pt_cld[:, :3] # 提取位置 [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # 初始旋转设为单位四元数 [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda") # 初始不透明度设为 0 (激活后为 0.5)
    
    params = {
        'means3D': means3D, # 位置 X_i
        'rgb_colors': init_pt_cld[:, 3:6], # 颜色 C_i
        'unnorm_rotations': unnorm_rots, # 四元数表示的旋转矩阵 R_i
        'logit_opacities': logit_opacities, # 不透明度 λ_i
        # 缩放初始化为邻域距离的对数；尺度矩阵 S_i
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1 if use_simplification else 2)),
    }
    
    if not use_simplification:
        # 如果不简化，增加球谐函数的高阶项
        params['feature_rest'] = torch.zeros(num_pts, 45) # 设定 SH degree 3

    # 初始化相机姿态轨迹（相对第一帧）
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots # 相机旋转轨迹
    params['cam_trans'] = np.zeros((1, 3, num_frames)) # 相机平移轨迹

    # 将所有 numpy 数组转换为可训练的 PyTorch 参数，并移动到 GPU
    for k, v in params.items():
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    # 初始化辅助变量
    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(), # 最大 2D 半径
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(), # 梯度累积
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(), # 归一化分母
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()} # 创建点的时间戳

    return params, variables
```

在获得优化后的二维高斯表示以及给定位姿$ \mathbf{P} $后，我们采用 _[39]_ 中提出的光栅化方法进行新视角渲染。_UV 空间 (UV space) _与世界坐标空间之间的变换可表示为：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767599239686-5499c01e-7056-4989-aa97-858a47fffb92.png)

```python
# utils\slam_helpers.py L238-L273
def transform_to_frame(params, time_idx, gaussians_grad, camera_grad):
    """
    将 Gaussian 中心从世界坐标系变换到指定的相机帧坐标系。
    输入:
        params: 包含 Gaussian 和相机姿态参数的字典
        time_idx: 目标帧的时间索引
        gaussians_grad: 是否允许 Gaussian 点云位置的梯度
        camera_grad: 是否允许相机姿态的梯度
    作用:
        获取指定帧的相机旋转和平移，构建世界到相机的变换矩阵，并将所有 3D 点变换到该相机坐标系下。
    输出:
        transformed_pts: 变换后的 Gaussian 中心点
        rel_w2c: 世界到相机的变换矩阵
    """
    # 获取对应帧的相机姿态
    if camera_grad:
        # 允许梯度时，直接使用参数并归一化旋转
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx])
        cam_tran = params['cam_trans'][..., time_idx]
    else:
        # 不允许梯度时，使用 detach() 切断计算图
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        cam_tran = params['cam_trans'][..., time_idx].detach()
    
    # 构建 4x4 的外参矩阵 (w2c)
    rel_w2c = torch.eye(4).cuda().float()
    rel_w2c[:3, :3] = build_rotation(cam_rot)  # 填充旋转部分 R_i
    rel_w2c[:3, 3] = cam_tran                  # 填充平移部分 

    # 获取世界坐标系下的点云
    if gaussians_grad:
        pts = params['means3D']
    else:
        pts = params['means3D'].detach()
    
    # 变换点云到相机坐标系
    pts_ones = torch.ones(pts.shape[0], 1).cuda().float()  # 齐次坐标的 1
    pts4 = torch.cat((pts, pts_ones), dim=1)               # 拼接成 (N, 4)
    transformed_pts = (rel_w2c @ pts4.T).T[:, :3]           # 矩阵乘法并取前 3 维

    return transformed_pts, rel_w2c
```

结合由位姿$ \mathbf{P} $给出的世界到_屏幕 (screen) _的变换$ \mathbf{W} $，对于屏幕上的一个像素点$ \mathbf{x}=(x,y) $，其对应的射线满足$ 
(xz,yz,z,1){\top}=\mathbf{W}\mathbf{H}(u,v,1,1){\top} $，其中$ z $表示射线与高斯溅射的交点深度。在光栅化过程中 _[39]_，我们通过如下方式求解屏幕点$ \mathbf{x} $在_高斯局部坐标系 _下的交点$ \mathbf{u}(\mathbf{x})=(u(\mathbf{x}),v(\mathbf{x})) $：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767599326808-880fe9d8-8e69-475b-9d41-fbbfaa693d0a.png)

其中$ \mathbf{h}_u=(\mathbf{W}\mathbf{H})^{\top}\mathbf{h}_x $、$ \mathbf{h}_v=(\mathbf{W}\mathbf{H})^{\top}\mathbf{h}_y $为两个四维齐次平面，$ h^i_u,h^i_v $表示对应向量的第$ i $个分量。

<details class="lake-collapse"><summary id="udd016593"><em><span class="ne-text">UV 空间 (UV space)/屏幕 (screen) </span></em><span class="ne-text">：</span><span class="ne-text" style="color: rgb(51, 51, 51); text-decoration: underline">UV 空间</span><span class="ne-text" style="color: rgb(51, 51, 51)">是每个 2D 高斯的局部坐标系，</span><span class="ne-text" style="color: rgb(51, 51, 51); text-decoration: underline">屏幕空间</span><span class="ne-text" style="color: rgb(51, 51, 51)">是最终的 2D 图像输出平面，用于显示渲染结果。（</span><em><span class="ne-text">高斯局部坐标系</span></em><span class="ne-text"> = UV 空间</span><span class="ne-text" style="color: rgb(51, 51, 51)">）</span></summary><p id="uc9a8f1a1" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/45861457/1768114838400-57c6e4c7-7d51-4e2d-9cdf-559197803041.png" width="546.1512429966248" title="" crop="0,0,1,1" id="udacce5d3" class="ne-image"></p></details>
最终的像素颜色$ c(\mathbf{x}) $由以下方式渲染得到：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767599401239-86bdb677-c04d-4225-a71e-c7ba884177e3.png)

其中$ 
T_i=\prod_{j=1}^{i-1}\bigl(1-\alpha_j\hat{G}_j(\mathbf{u}(\mathbf{x}))\bigr)
 $表示第$ i $个二维高斯的可见性，$ \mathbf{c}_i $与$ \alpha_i $分别为其对应的 RGB 颜色和不透明度，$ \hat{G}_i $表示按照 _[39]_ 中对象空间低通滤波处理后的二维高斯值。

在给定位姿下，对应的深度图渲染为

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767599488969-82cac632-6ba2-41ca-86dc-af9a7ab0d2e3.png)

其中$ \omega_i=T_i\alpha_i\hat{G}_i(\mathbf{u}(\mathbf{x})) $表示第$ i $个高斯对该像素的权重贡献，$ z_i $为对应交点的深度值，$ \varepsilon $为防止分母为零而引入的极小常数。

```python
# scripts\main.py L70-L101

# RGB 渲染过程
rendervar['means2D'].retain_grad()  # 保留 2D 中心梯度用于密度化
## GaussianRasterizer 内部实现公式(3)的UV交点计算（调用的 2DGS 光栅化库）
im, radius, allmap = Renderer(raster_settings=curr_data['cam'])(**rendervar)

# 应用曝光参数修正 (exp_a, exp_b)
exp_a = curr_data['exp']['exp_a']
exp_b = curr_data['exp']['exp_b']
im = im*torch.exp(exp_a)+exp_b
im = torch.clamp(im, 0, 1) # 限制在 0-1 范围

# RGB 渲染过程
rendervar['means2D'].retain_grad()  # 保留 2D 中心梯度用于密度化
im, radius, allmap = Renderer(raster_settings=curr_data['cam'])(**rendervar)

# 应用曝光参数修正 (exp_a, exp_b)
exp_a = curr_data['exp']['exp_a']
exp_b = curr_data['exp']['exp_b']
im = im*torch.exp(exp_a)+exp_b
im = torch.clamp(im, 0, 1) # 限制在 0-1 范围

variables['means2D'] = rendervar['means2D'] # 缓存 2D 位置
depth = allmap[0:1] # 获取渲染深度
render_alpha = allmap[1:2] # 获取渲染透明度 (Alpha/Silhouette)

# 生成有效像素掩码
presence_sil_mask = (render_alpha > sil_thres) # 覆盖区域掩码

nan_mask = (~torch.isnan(depth)) # 排除无效深度
bg_mask = energy_mask(curr_data['im']) # 能量掩码（排除边缘或特定的低能量区域）

# 深度异常值过滤
if ignore_outlier_depth_loss and use_dep:
    depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
    mask = (depth_error < 20*depth_error.mean())
    mask = mask & (curr_data['depth'] > 0)
else:
    mask = (curr_data['depth'] > 0)
mask = mask & nan_mask & bg_mask

# 如果是追踪或 BA 模式，通常只在有 Gaussian 覆盖的区域计算损失
if (tracking or do_ba) and use_sil_for_loss:
    mask = mask & presence_sil_mask
    
# 法线渲染与坐标变换（从观察空间转回世界空间）
render_normal = allmap[2:5] # 获取渲染法向量
render_normal = (render_normal.permute(1,2,0) @ 
                 (w2c[:3,:3].T)).permute(2,0,1)
```

| **变量** | **含义** | **数据类型** | **用途** |
| --- | --- | --- | --- |
| `**<font style="color:rgb(51, 51, 51);">im</font>**` | <font style="color:rgb(51, 51, 51);">RGB渲染图像</font> | `**<font style="color:rgb(51, 51, 51);">[H, W, 3]</font>**`<font style="color:rgb(51, 51, 51);"> 张量</font> | <font style="color:rgb(51, 51, 51);">最终颜色输出，用于显示和损失计算</font> |
| `**<font style="color:rgb(51, 51, 51);">radius</font>**` | <font style="color:rgb(51, 51, 51);">2D 高斯屏幕半径</font> | `**<font style="color:rgb(51, 51, 51);">[N]</font>**`<font style="color:rgb(51, 51, 51);"> 张量</font> | <font style="color:rgb(51, 51, 51);">跟踪高斯在屏幕上的投影大小，用于稠密化</font> |
| `**<font style="color:rgb(51, 51, 51);">allmap</font>**` | <font style="color:rgb(51, 51, 51);">附加渲染图集</font> | `**<font style="color:rgb(51, 51, 51);">[5, H, W]</font>**`<font style="color:rgb(51, 51, 51);"> 张量</font> | —— |


### **跟踪（Tracking）**
与以往工作 _[37]_ 类似，我们在新帧位姿初始化时假设相机满足**恒定速度模型**。给定前两帧位姿$ \mathbf{P}_{t-1} $和$ \mathbf{P}_{t-2} $，首先计算位姿变化量$ \Delta(\mathbf{P}_{t-1},\mathbf{P}_{t-2}) $，并据此初始化当前帧位姿：$ \mathbf{P}_t = \mathbf{P}_{t-1} + \Delta(\mathbf{P}_{t-1},\mathbf{P}_{t-2}) $随后，通过梯度下降方式对相机位姿进行迭代优化。

<details class="lake-collapse"><summary id="u99e92958"><em><span class="ne-text">[37]</span></em></summary><p id="ue2379d2c" class="ne-p"><span class="ne-text">[37] </span><strong><span class="ne-text">SplaTAM: Splat, Track &amp; Map 3D Gaussians for Dense RGB-D SLAM（CVPR 2024）</span></strong><span class="ne-text">：面向 RGB-D 场景提出基于 3D 高斯的跟踪与建图一体化框架，充分利用深度观测约束提升位姿估计精度与尺度稳定性，实现高效、鲁棒的稠密 SLAM，为 3D Gaussian Splatting 在多传感器 SLAM 中的应用提供了重要实践。</span></p></details>
为实现几何上更加鲁棒的跟踪，本文提出一种**面向表面法向的跟踪正则项**，同时结合投影的点到点距离$ d_{\text{point-to-point}}(\mathbf{x}) $ 与点到平面距离$ d_{\text{point-to-plane}}(\mathbf{x}) $。设$ \mathbf{x} $为由像素$ \mathbf{x} $投影得到的三维点，$ \mathbf{x}_{GT} $表示由真实深度获得的三维点。点到点距离定义为  
<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767599725244-13e49a17-2ada-4218-ad67-03aacee28d49.png)

其中$ z(\cdot) $表示深度值，$ \mathbf{D}_{GT} $与$ \mathbf{D} $分别为真实深度和渲染深度。

```python
# scripts\main.py L102-L108;L125-L130

# curr_data: 当前帧数据（RGB图像、深度图、相机参数等），作为函数 get_loss 输入
if use_dep:
    # 从真值深度图计算真值法线和点云
    real_normal, real_points = depth_to_normal(curr_data['cam'], 
                                curr_data['depth'], 
                                gaussians_grad, 
                                camera_grad)
    
    real_normal = real_normal.permute(2,0,1) * (render_alpha).detach()
    
    if tracking or do_ba:
        # 渲染点云（基于渲染深度）
        render_points = depths_to_points(curr_data['cam'], 
                                    depth, 
                                    gaussians_grad, 
                                    camera_grad)
...

# 深度损失 L1 计算：curr_data['depth'] 真实深度 D_{GT}，depth 预测深度 D
if use_l1 and use_dep:
    mask = mask.detach()
    if tracking:
        losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
    else: # mapping 使用均值 L1
        losses['depth'] = torch.abs(curr_data['depth'] - depth).mean()
```

由于仅使用透视点到点距离无法充分利用表面几何信息，本文进一步引入**法向感知的点到平面距离**：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767599854502-385ff5b7-5966-49ef-998c-e9c2dda9ebaf.png)

其中真实表面法向$ \mathbf{N}_{GT} $通过相邻点的有限差分计算得到：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767599872024-5f24cfb5-48df-4c2f-bc0e-9abb78df9c72.png)

这里，$ \nabla_x $与$ \nabla_y $分别表示沿$ x $方向和$ y $方向的差分算子。

```python
# scripts\main.py L136-L144

# 点到面（Point-to-Plane）损失，常用于提高姿态估计精度
## render_points: x; real_points: x_{GT}; normal_real_vec: N_{GT}
if (tracking or do_ba) and use_normal:
    mask_point = mask.reshape(-1)
    normal_real_vec = real_normal.reshape(-1, 3)[mask_point][..., None]
    point_err = (render_points.reshape(-1, 3)[mask_point] - real_points.reshape(-1, 3)[mask_point])[:, None, :]
    real_dir_dist = torch.bmm(point_err, normal_real_vec) # 投影到真值法线上
    losses['point2plane'] = torch.abs(real_dir_dist).sum()
```

综合外观一致性约束，本文将<u>最终的跟踪损失函数</u>定义为：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767599900265-88d7ac37-0fdd-4304-9bbf-4af0a1ea6e53.png)

其中$ \mathcal{L}_c
= \lVert \hat{\mathbf{C}}(\mathbf{x}) - \mathbf{C}_{GT}(\mathbf{x}) \rVert_1
 $表示渲染图像与真实图像之间的$ L_1 $损失，$ \hat{\mathbf{C}} $为经过仿射曝光校正 _[35]_ 后的渲染颜色，$ \mathbf{C}_{GT} $为真实颜色。

<details class="lake-collapse"><summary id="u69d893e9"><em><span class="ne-text">[35]</span></em></summary><p id="u4a5e6f2e" class="ne-p"><span class="ne-text">[35] </span><strong><span class="ne-text">Gaussian Splatting SLAM（CVPR 2024）</span></strong><span class="ne-text">：首次系统性地将 3D Gaussian Splatting 引入 SLAM 框架，通过以高斯作为显式可微场景表示，实现相机位姿估计与高质量稠密建图的联合优化，在保持实时渲染能力的同时显著提升了地图的几何与外观表达能力，标志着 SLAM 从隐式神经场向高效显式神经表示的重要转变。</span></p></details>
```python
# scripts\main.py L147-L160

# RGB 图像损失
if (tracking or do_ba) and (use_sil_for_loss or ignore_outlier_depth_loss):
    color_mask = torch.tile(mask, (3, 1, 1))
    color_mask = color_mask.detach()
    losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    
elif tracking or do_ba:
    losses['im'] = torch.abs(curr_data['im'] - im).sum()
else: # Mapping 模式：混合 L1 和 SSIM 损失，追求更高的图像保真度
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

# 应用权重计算总损失
weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
loss = sum(weighted_losses.values())
```

### 高斯扩展与关键帧采样（Gaussian Expanding and Keyframe Sampling）
为在相机跟踪与建图过程中保持稳定的优化行为，本文采用**高斯扩展策略** _[37]_。<u>具体而言，我们将当前输入帧加入关键帧候选列表，并每隔</u>$ k $<u>帧更新一次二维高斯场景表示，以逐步引入此前未观测到的组织区域。在二维高斯更新过程中，我们利用渲染得到的轮廓图（silhouette map）</u>$ S(\mathbf{x})=\frac{\sum_i \omega_i}{\sum_i \omega_i + \varepsilon} $<u>来表示当前已观测区域。对于轮廓值低于阈值</u>$ \rho_e $<u>（即</u>$ S(\mathbf{x}) < \rho_e $<u>）的区域，以及真实几何位于已重建组织表面之上的区域，我们对二维高斯表示进行扩展。新增的高斯基元使用当前输入图像与深度信息进行初始化</u>。

<details class="lake-collapse"><summary id="u71bdbf4e"><em><span class="ne-text">[37]</span></em></summary><p id="u68a1fe0b" class="ne-p"><span class="ne-text">[37] </span><strong><span class="ne-text">SplaTAM: Splat, Track &amp; Map 3D Gaussians for Dense RGB-D SLAM（CVPR 2024）</span></strong><span class="ne-text">：面向 RGB-D 场景提出基于 3D 高斯的跟踪与建图一体化框架，充分利用深度观测约束提升位姿估计精度与尺度稳定性，实现高效、鲁棒的稠密 SLAM，为 3D Gaussian Splatting 在多传感器 SLAM 中的应用提供了重要实践。</span></p></details>
```python
# utils\slam_helpers.py L315-L364
def add_new_gaussians(params, variables, curr_data, sil_thres, time_idx, mean_sq_dist_method, use_simplification=True, config=None):
    """
    在渲染不完整的地方添加新的 Gaussian。
    输入:
        params: 当前 Gaussian 参数
        variables: 辅助变量（如梯度累积等）
        curr_data: 当前帧的观测数据（图像、深度、相机等）
        sil_thres: 轮廓阈值，低于此值认为该区域未被 Gaussian 覆盖
        time_idx: 当前帧索引
        mean_sq_dist_method: 计算平均平法距离的方法
        use_simplification: 是否使用简化模型
        config: 配置字典
    作用:
        通过渲染当前视角下的轮廓图，找出未覆盖区域或深度误差较大的区域，并从真实深度图中反求出 3D 点来初始化并添加新的 Gaussian 元素。
    输出:
        params: 更新后的参数字典
        variables: 更新后的辅助变量字典
    """
    use_dep = config['use_dep']  # 是否使用深度信息
    # 渲染当前视角下的轮廓（Silhouette）
    transformed_pts, _ = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    rendervar = transformed_params2rendervar(params, transformed_pts)
    im, _, allmap = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    silhouette = allmap[1:2]  # 获取渲染的轮廓图
    non_presence_sil_mask = (silhouette < sil_thres)  # 找到未覆盖区域
    
    # 如果允许使用深度，进一步检查前景色缺失（即渲染深度远大于真实深度的地方）
    if use_dep:
        gt_depth = curr_data['depth'][0, :, :]  # 获取真值深度
        render_depth = allmap[0:1]              # 获取渲染深度
        depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
        # 认为渲染深度过度落后且误差巨大的地方需要添加
        non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 20*depth_error.mean())
        non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    else:
        non_presence_mask = non_presence_sil_mask
        
    # 展平掩码以便索引
    non_presence_mask = non_presence_mask.reshape(-1)

    # 如果有需要添加新 Gaussian 的像素
    if torch.sum(non_presence_mask) > 0:
        # 获取当前帧的相机世界坐标 (W2C)
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        
        # 结合深度 valid 掩码和图像能量掩码
        if use_dep:
            valid_depth_mask = (curr_data['depth'][0, :, :] > 0) & (curr_data['depth'][0, :, :] < 1e10)
            non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        
        valid_color_mask = energy_mask(curr_data['im']).squeeze()
        non_presence_mask = non_presence_mask & valid_color_mask.reshape(-1)
        
        # 从掩码区域的图像和深度生成新的点云
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        
        # 初始化这些新点的参数并拼接到全局参数中
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, use_simplification)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
            
        # 更新辅助变量（为新点补充全零梯度累积器等）
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        
        # 为新点标记它们产生的时间索引
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'], new_timestep),dim=0)
        
    return params, variables
```

<u>在完成高斯扩展后，我们从此前的候选关键帧列表中采样</u>$ n $<u>帧，用于后续的建图与束调整</u>。为此，本文提出一种**位姿一致的关键帧采样策略**，同时考虑平移与旋转误差。受文献 _[38]_ 启发，我们为每一个候选帧定义其采样概率为：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767600055257-50d911e4-c775-43a8-a930-15892a94c6a1.png)

 其中$ d_l $表示位姿位置的$ L_2 $距离，$ r_l $为旋转四元数的幅值误差，$ t_l $为相对于当前帧的时间间隔，$ s $为尺度常数，本文中设为$ 0.2 $。随后，对所有候选帧的概率进行归一化，使其满足$ \sum_i P_i = 1 - P_c $，其中$ P_c $为当前帧的预设采样概率。最终，所有候选帧根据其概率的累积分布函数进行排序并完成采样。

<details class="lake-collapse"><summary id="u32bc0bc6"><em><span class="ne-text">[38]</span></em><em><span class="ne-text" style="color: #DF2A3F">*</span></em></summary><p id="u3fbd7e8e" class="ne-p"><span class="ne-text">[38] </span><strong><span class="ne-text">EndoGSLAM</span></strong><strong><span class="ne-text" style="color: #DF2A3F">*</span></strong><strong><span class="ne-text">: Real-Time Dense Reconstruction and Tracking in Endoscopic Surgeries Using Gaussian Splatting（arXiv 2024）</span></strong><span class="ne-text">：提出面向内窥镜手术场景的实时 SLAM 框架，将 3D Gaussian Splatting 引入内窥镜位姿跟踪与稠密重建过程，通过高斯显式表示联合优化相机位姿与场景几何，在弱纹理、强反光和软组织形变等复杂条件下实现高质量重建与稳定跟踪，是 3DGS-SLAM 在内窥镜手术应用中的代表性工作。  </span></p></details>
```python
def keyframe_selection_ape(time_idx, curr_position, curr_rotation, keyframe_list, distance_current_frame_prob, n_samples):
    """
    作用: 在距离和时间基础上增加旋转差异（APE 风格）权重，建立概率分布并采样关键帧。

    输入:
        time_idx (int): 当前时间步.
        curr_position (array): 当前相机位置.
        curr_rotation (tensor): 当前相机旋转参数（用于 build_rotation）.
        keyframe_list (list): 历史关键帧列表.
        distance_current_frame_prob (float): 采样当前帧的概率.
        n_samples (int): 采样数量.

    输出:
        sample_indices (list): 采样选中的关键帧索引列表.
    """
    distances = []
    rot_err = []
    time_laps = []
    # 计算当前旋转和位置的量级作为缩放因子
    curr_rot = torch.linalg.norm(curr_rotation)
    curr_shift = np.linalg.norm(curr_position)
    for keyframe in keyframe_list:
        est_w2c = keyframe['est_w2c'].detach().cpu()
        camera_position = est_w2c[:3, 3]
        distance = np.linalg.norm(camera_position - curr_position) # 距离误差
        time_lap = time_idx - keyframe['id'] # 时间间隔
        distances.append(distance)
        time_laps.append(time_lap)
        
        # 计算当前帧与关键帧之间的旋转差异（四元数模长表示）
        rot_curr = build_rotation(curr_rotation)[0]
        rot_est = est_w2c[:3, :3].detach().cpu()
        # 计算相对旋转矩阵并转为四元数，取模长作为旋转误差
        err_quat = matrix_to_quaternion((torch.linalg.inv(rot_est)@rot_curr.cpu())[None])
        rot_err.append(np.linalg.norm(err_quat)) # 旋转误差

    # 概率转换函数
    def dis2prob(x, scaler):
        return np.log2((1 + scaler/(x+scaler/5)))
    
    # 综合考虑距离误差、时间跨度和旋转误差
    dis_prob = [dis2prob(d, curr_shift)+dis2prob(t, time_idx)+dis2prob(r, curr_rot)
                for d, t, r in zip(distances, time_laps, rot_err)]
    # 归一化概率分布
    sum_prob = sum(dis_prob) / (1-distance_current_frame_prob)
    norm_dis_prob = [p/sum_prob for p in dis_prob]
    # 添加当前帧概率
    norm_dis_prob.append(distance_current_frame_prob)
    
    # 根据 CDF 进行逆变换采样
    cdf = np.cumsum(norm_dis_prob)
    samples = np.random.rand(n_samples)
    sample_indices = np.searchsorted(cdf, samples)

    return sample_indices
```

### **建图（Mapping）**
<u>在跟踪模块输出优化后的相机位姿基础上，我们在</u>**<u>固定相机位姿</u>**<u>的条件下，采用梯度下降方法对二维高斯场景表示进行迭代更新</u>。为实现外观紧凑且深度一致的二维高斯表示优化，本文遵循文献 _[39]_，引入**深度畸变损失（depth distortion loss）**，以最小化同一射线上不同高斯溅射交点之间的深度差异：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767600209732-1c0bd5f8-9db2-4dce-b6e6-fb56b627f3f5.png)

其中$ \omega_i $和$ z_i $分别表示第$ i $个交点对应的权重和深度值。

<details class="lake-collapse"><summary id="ube052754"><em><span class="ne-text">[39]</span></em><em><span class="ne-text" style="color: #DF2A3F">*</span></em></summary><p id="u4a13341f" class="ne-p"><span class="ne-text">[39] </span><strong><span class="ne-text">2D Gaussian Splatting for Geometrically Accurate Radiance Fields（SIGGRAPH 2024）</span></strong><strong><span class="ne-text" style="color: #DF2A3F">*</span></strong><span class="ne-text">：提出以二维高斯为基本表示的可微渲染框架，通过在像平面中进行高斯溅射并引入严格的几何一致性约束，实现对辐射场的高精度建模，在几何准确性与渲染效率之间取得良好平衡，为高斯溅射表示在高精度几何重建与实时渲染中的进一步发展提供了新的理论与方法基础。  </span></p></details>
受 _[39]_ 启发，我们进一步通过**法向一致性损失（normal consistency loss）**对二维高斯的表面法向进行约束：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767600256657-cfe440f9-a072-4f8b-aa4b-e2f0684d2522.png)

其中$ i $表示沿同一射线被命中的高斯索引，$ \mathbf{n}_i $为对应高斯相对于相机坐标系的表面法向。

综合上述约束，本文将最终的建图损失函数定义为：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767600436345-66b89099-ff6c-4145-84f9-00f8e3e970eb.png)

其中$ \mathcal{L}_{rec}
= (1-\lambda)\lVert \hat{\mathbf{C}}(\mathbf{x}) - \mathbf{C}_{GT}(\mathbf{x}) \rVert_1
+ \lambda\mathcal{L}_{\text{D-SSIM}} $为颜色重建损失，$ \mathcal{L}_{\text{D-SSIM}} $表示文献 _[34]_ 中采用的 SSIM 损失。本文中参数取值为：$ \lambda=0.2，\alpha=1000，\beta=0.05 $，与 _[39]_ 保持一致。

<details class="lake-collapse"><summary id="ua05392fe"><em><span class="ne-text">[34]</span></em><em><span class="ne-text" style="color: #DF2A3F">*</span></em></summary><p id="u58bbd911" class="ne-p"><span class="ne-text">[34] </span><strong><span class="ne-text">3D Gaussian Splatting for Real-Time Radiance Field Rendering（ACM TOG 2023）</span></strong><strong><span class="ne-text" style="color: #DF2A3F">*</span></strong><span class="ne-text">：提出以各向异性 3D 高斯作为显式场景表示的新型辐射场建模方法，通过高斯投影与可微 splatting 实现无需体采样的高效渲染，在保持高质量新视角合成效果的同时显著提升实时性能，为后续基于 Gaussian 表示的实时重建与 SLAM 方法奠定了核心表示与渲染框架基础。  </span></p></details>
```python
# scripts\main.py L

# 建图过程中的法线损失 L_n
if (mapping or do_ba) and use_dep and use_normal:
    normal_error = ((1 - (render_normal * real_normal).sum(dim=0))[None]).mean()
    normal_loss = config['mapping']['lambda_normal'] * (normal_error)
    losses['normal'] = normal_loss
...
        
if mapping: # L_d
    # 深度分布正则化或相关项
    render_dist = allmap[6:7]
    losses['depth_dist'] = render_dist.mean()
...
    
# RGB 图像损失
if (tracking or do_ba) and (use_sil_for_loss or ignore_outlier_depth_loss):
    color_mask = torch.tile(mask, (3, 1, 1))
    color_mask = color_mask.detach()
    losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
elif tracking or do_ba:
    losses['im'] = torch.abs(curr_data['im'] - im).sum()
else: # Mapping 模式：混合 L1 和 SSIM 损失，追求更高的图像保真度；L_{rec}
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

# 应用权重计算总损失
## α 和 β 在 loss_weights 中
weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
loss = sum(weighted_losses.values())

if mapping:
    # 更新被看到的 Gaussian 的最大 2D 半径
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
weighted_losses['loss'] = loss

return loss, variables, weighted_losses
```

### 束调整（Bundle Adjustment）
<u>在获得一系列相机位姿以及经建图模块优化后的二维高斯场景表示后，本文采用</u>**<u>基于梯度下降的束调整（BA）方法</u>**<u>，对相机位姿与场景表示进行联合优化</u>。<u>具体而言，我们每进行 100 次跟踪迭代便执行一次 BA，从关键帧候选列表中选取若干关键帧参与优化。</u>

<u>关键帧选择策略遵循 </u>[<u>1.3.3</u>](#Gnm5l)<u> 节中提出的方法，并对选中的关键帧进行随机排列，在总计 200 次迭代中交替更新相机位姿与二维高斯参数</u>。给定相机位姿 $ \mathbf{P}={\mathbf{R},\mathbf{t}} $ 以及二维高斯集合 $ \mathcal{G} $，束调整通过最小化如下损失函数实现：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767600525213-c3d62c47-4c79-4093-8d7d-aafddc9447ad.png)

```python
# scripts\main.py L96-L161

# 如果是追踪或 BA 模式，通常只在有 Gaussian 覆盖的区域计算损失
if (tracking or do_ba) and use_sil_for_loss:
    mask = mask & presence_sil_mask
    
# 法线渲染与坐标变换（从观察空间转回世界空间）
render_normal = allmap[2:5]
render_normal = (render_normal.permute(1,2,0) @ 
                 (w2c[:3,:3].T)).permute(2,0,1)

if use_dep:
    # 从真值深度图计算真值法线和点云
    real_normal, real_points = depth_to_normal(curr_data['cam'], 
                                curr_data['depth'], 
                                gaussians_grad, 
                                camera_grad)
    
    real_normal = real_normal.permute(2,0,1) * (render_alpha).detach()
    
    if tracking or do_ba:
        # 渲染点云（基于渲染深度）
        render_points = depths_to_points(curr_data['cam'], 
                                    depth, 
                                    gaussians_grad, 
                                    camera_grad)
  
# 建图过程中的法线损失
if (mapping or do_ba) and use_dep and use_normal:
    normal_error = ((1 - (render_normal * real_normal).sum(dim=0))[None]).mean()
    normal_loss = config['mapping']['lambda_normal'] * (normal_error)
    losses['normal'] = normal_loss

# 深度损失 L1 计算
if use_l1 and use_dep:
    mask = mask.detach()
    if tracking:
        losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
    else: # mapping 使用均值 L1
        losses['depth'] = torch.abs(curr_data['depth'] - depth).mean()
        
if mapping:
    # 深度分布正则化或相关项
    render_dist = allmap[6:7]
    losses['depth_dist'] = render_dist.mean()
    
# 点到面（Point-to-Plane）损失，常用于提高姿态估计精度
if (tracking or do_ba) and use_normal:
    mask_point = mask.reshape(-1)
    normal_real_vec = real_normal.reshape(-1, 3)[mask_point][..., None]
    point_err = (render_points.reshape(-1, 3)[mask_point] - real_points.reshape(-1, 3)[mask_point])[:, None, :]
    real_dir_dist = torch.bmm(point_err, normal_real_vec) # 投影到真值法线上
    losses['point2plane'] = torch.abs(real_dir_dist).sum()
    
# RGB 图像损失
if (tracking or do_ba) and (use_sil_for_loss or ignore_outlier_depth_loss):
    color_mask = torch.tile(mask, (3, 1, 1))
    color_mask = color_mask.detach()
    losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    
elif tracking or do_ba:
    losses['im'] = torch.abs(curr_data['im'] - im).sum()
else: # Mapping 模式：混合 L1 和 SSIM 损失，追求更高的图像保真度
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

# 应用权重计算总损失
weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
loss = sum(weighted_losses.values())
```

## EXPERIMENT
在实验部分中，我们在公开的内窥镜 SLAM 数据集上，对所提出的 <font style="background-color:#F297CC;">Endo-2DTAM</font> 在多种体内手术场景下的性能进行了评估。通过与当前最先进的 SLAM 方法进行对比实验，结果表明本文方法在内窥镜场景重建任务中展现出更为优越的性能。

### **实验设置（Experiment Setup）**
**A. 实验设置（Experiment Setup）**

1. **数据集（Dataset）**：为了进行评估，我们在结肠镜三维视频数据集（Colonoscopy 3D Video Dataset, <font style="background-color:#F8B881;">C3VD</font>）上进行实验。遵循文献 _[38]_，我们选择了 10 段分辨率为 675 × 540 的视频片段，每段视频平均包含 638 帧。

<details class="lake-collapse"><summary id="u1811329c"><em><span class="ne-text">[38]</span></em><em><span class="ne-text" style="color: #DF2A3F">*</span></em></summary><p id="u05d7490d" class="ne-p"><span class="ne-text">[38] </span><strong><span class="ne-text">EndoGSLAM</span></strong><strong><span class="ne-text" style="color: #DF2A3F">*</span></strong><strong><span class="ne-text">: Real-Time Dense Reconstruction and Tracking in Endoscopic Surgeries Using Gaussian Splatting（arXiv 2024）</span></strong><span class="ne-text">：提出面向内窥镜手术场景的实时 SLAM 框架，将 3D Gaussian Splatting 引入内窥镜位姿跟踪与稠密重建过程，通过高斯显式表示联合优化相机位姿与场景几何，在弱纹理、强反光和软组织形变等复杂条件下实现高质量重建与稳定跟踪，是 3DGS-SLAM 在内窥镜手术应用中的代表性工作。  </span></p></details>
2. **基线方法（Baselines）**：我们将本文方法与以下方法进行对比：传统方法 <font style="background-color:#FCE75A;">ORB-SLAM3</font>、隐式神经网络方法 <font style="background-color:#FCE75A;">NICE-SLAM</font> _[31]_、内窥镜 SLAM 方法 <font style="background-color:#FCE75A;">EndoDepth</font> _[18]_，以及最先进的基于 3DGS 的内窥镜 SLAM 方法 <font style="background-color:#FCE75A;">EndoGSLAM</font> _[38]_。为保证公平性，所有基线方法均使用 RGB-D 输入。

<details class="lake-collapse"><summary id="ubc4fe26f"><em><span class="ne-text">[31]、[18]</span></em></summary><p id="u5c468341" class="ne-p"><span class="ne-text">[31] </span><strong><span class="ne-text">NICE-SLAM</span></strong><strong><span class="ne-text" style="color: #DF2A3F">*</span></strong><strong><span class="ne-text">: Neural Implicit Scalable Encoding for SLAM（CVPR 2022）</span></strong><span class="ne-text">：通过多分辨率层级隐式编码结构，将局部与全局几何信息解耦表示，有效提升了神经 SLAM 在大规模场景中的可扩展性和重建质量，是隐式神经表示与 SLAM 结合的重要代表性工作。</span></p><p id="u63169a30" class="ne-p"><span class="ne-text">[18] </span><strong><span class="ne-text">EndoDepth-and-Motion: Reconstruction and Tracking in Endoscopic Videos Using Depth Networks and Photometric Constraints（RA-L 2021）</span></strong><span class="ne-text">：将单目深度学习网络与经典光度一致性约束相结合，实现端到端的内窥镜位姿跟踪与三维重建，在无需真实深度标注的情况下获得较为稳定的深度与运动估计，体现了“深度学习 + 几何约束”的混合建模思想。</span></p></details>
3. **评价指标（Metrics）**：在相机轨迹评估中，我们报告绝对轨迹误差（Absolute Trajectory Error, ATE, mm）。在深度评估中，我们使用均方根误差（Root Mean Square Error, RMSE, mm）。此外，对于新视角合成，我们还报告标准光度渲染质量指标（PSNR、SSIM 和 LPIPS）。
4. **实现细节（Implementation Details）**：所有实验均在 **Core i7-13700K CPU、RTX 4090 GPU、Ubuntu 20.04** 环境下进行。我们的 SLAM 系统实现了三个版本：**Endo-2DTAM-Base**、**Endo-2DTAM-Small** 和 **Endo-2DTAM-Tiny**。
+ **Endo-2DTAM-Base**：关键帧概率$ p_c=0.1
 $，关键帧候选列表每 8 帧更新一次；跟踪和建图每帧各 15 次迭代。
+ **Endo-2DTAM-Small**：跟踪每帧 10 次迭代，建图每 2 帧 10 次迭代，采用 1/2 分辨率。
+ **Endo-2DTAM-Tiny**：跟踪每帧 8 次迭代，建图每 2 帧 8 次迭代，采用 1/4 分辨率。

对于 Endo-2DTAM-Small 和 Endo-2DTAM-Tiny，关键帧概率$ p_c $设置为 0.5，关键帧候选列表每 4 帧更新一次。三个版本均采用$ \rho_e=0.5 $。

### **实验结果（Experiment Results）**
<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767600892003-67ca275e-2dc6-4e33-977b-f16e795ac3f0.png)

**表 1：C3VD 数据集的定量分析结果**

我们在 C3VD 数据集上评估了本文方法的三个版本，结果如_表 1 _所示。遵循文献 _[38]_，我们将每个场景划分为训练集和测试集。我们从以下几个方面对 Endo-2DTAM 与其他基线方法进行比较：测试集中新视角渲染的外观与几何重建质量，以及训练集中所有帧的相机轨迹。

<details class="lake-collapse"><summary id="u5aae6f2d"><em><span class="ne-text">[38]</span></em><em><span class="ne-text" style="color: #DF2A3F">*</span></em></summary><p id="u075a248d" class="ne-p"><span class="ne-text">[38] </span><strong><span class="ne-text">EndoGSLAM</span></strong><strong><span class="ne-text" style="color: #DF2A3F">*</span></strong><strong><span class="ne-text">: Real-Time Dense Reconstruction and Tracking in Endoscopic Surgeries Using Gaussian Splatting（arXiv 2024）</span></strong><span class="ne-text">：提出面向内窥镜手术场景的实时 SLAM 框架，将 3D Gaussian Splatting 引入内窥镜位姿跟踪与稠密重建过程，通过高斯显式表示联合优化相机位姿与场景几何，在弱纹理、强反光和软组织形变等复杂条件下实现高质量重建与稳定跟踪，是 3DGS-SLAM 在内窥镜手术应用中的代表性工作。  </span></p></details>
在外观重建方面，本文方法实现了最高的 **SSIM 为 0.77 ± 0.07**，表明渲染结果最接近人类视觉感知。在深度重建方面，Endo-2DTAM-Base 显著优于所有其他基线方法，取得最低 **RMSE 为 1.87 ± 0.63 mm**。在保证高精度建图的同时，本方法在轨迹估计上也表现出竞争力，绝对轨迹误差 **ATE 为 0.33 ± 0.22 mm**。

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767600818407-363965dd-e97f-40ef-86f9-13236c7792ba.png)

**图 3. C3VD [56] 上的定性结果（Qualitative Result）**

我们将本文方法与最先进的 EndoGSLAM _[38]_ 进行了密集内窥镜 SLAM 的对比。从图中 cecum t2 b 和 sigmoid t2 a 的结果可以看出，本文方法在颜色和深度重建上更加稳健。同时，从 cecum t3 a 的轨迹结果可以看出，本方法的相机轨迹估计也更为精确。

定性结果如_图 3 _所示，也表明本文方法相比基于 3DGS 的方法，在渲染稳健性和轨迹估计精度上均具有优势。<u>与 EndoGSLAM-H 与 EndoGSLAM-R 之间存在较大性能差距不同，本文的 Endo-2DTAM-Small 与 Endo-2DTAM-Tiny 相对于 Endo-2DTAM-Base 的深度误差仅增加 </u>**<u>19.7%</u>**<u> 和 </u>**<u>27.8%</u>**<u>，轨迹误差的变化也较小，分别为 </u>**<u>15.5%</u>**<u> 和 </u>**<u>21.2%</u>**。

**这些结果表明，本文方法在不同分辨率下均能提供高精度几何重建和良好的鲁棒性，能够为手术重建任务提供可靠基础**，并支撑后续的外科机器人规划与导航等任务。

### **性能分析（Performance Analysis）**
<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767601011819-d09de62f-f94e-4875-904b-9ebf62bd587f.png)

**表 2：C3VD 数据集上的时效性**

如_表 2_ 所示，我们从三个方面对系统的运行效率进行了评估：**每帧的跟踪时间、每帧的建图时间，以及新视角的在线渲染速度**。<u>尽管 ORB-SLAM3 和 Endo-Depth</u>_<u> [18]</u>_<u> 在单帧处理的跟踪与建图速度上更快，但这两种方法依赖于后处理的体素融合来完成稠密场景重建，且无法实现在线新视角渲染</u>。<u>NICE-SLAM </u>_<u>[31]</u>_<u> 虽然在几何精度方面表现优异，但在实时跟踪与建图效率上仍存在明显不足</u>。

<details class="lake-collapse"><summary id="u532f2262"><em><span class="ne-text">[18] 、[31]</span></em></summary><p id="u65800bf8" class="ne-p"><span class="ne-text">[18] </span><strong><span class="ne-text">EndoDepth-and-Motion: Reconstruction and Tracking in Endoscopic Videos Using Depth Networks and Photometric Constraints（RA-L 2021）</span></strong><span class="ne-text">：将单目深度学习网络与经典光度一致性约束相结合，实现端到端的内窥镜位姿跟踪与三维重建，在无需真实深度标注的情况下获得较为稳定的深度与运动估计，体现了“深度学习 + 几何约束”的混合建模思想。</span></p><p id="uae6fd356" class="ne-p"><span class="ne-text">[31] </span><strong><span class="ne-text">NICE-SLAM</span></strong><strong><span class="ne-text" style="color: #DF2A3F">*</span></strong><strong><span class="ne-text">: Neural Implicit Scalable Encoding for SLAM（CVPR 2022）</span></strong><span class="ne-text">：通过多分辨率层级隐式编码结构，将局部与全局几何信息解耦表示，有效提升了神经 SLAM 在大规模场景中的可扩展性和重建质量，是隐式神经表示与 SLAM 结合的重要代表性工作。</span></p></details>
<u>与当前最先进的基于 3DGS 的 EndoGSLAM </u>_<u>[38]</u>_<u> 相比，Endo-2DTAM-Base 在保持具有竞争力的跟踪效率的同时，在建图阶段耗时更少</u>。此外，<u>本文提出的 Small 和 Tiny 版本相比 EndoGSLAM-R 具有更优的性能权衡，其运行速度分别达到 </u>**<u>5.53 fps</u>**<u> 和 </u>**<u>9.82 fps</u>**<u>，并且在跟踪精度和重建精度方面仍取得了更高的表现</u>。

<details class="lake-collapse"><summary id="u157020c8"><em><span class="ne-text">[38]</span></em><em><span class="ne-text" style="color: #DF2A3F">*</span></em></summary><p id="u3e9fb6bb" class="ne-p"><span class="ne-text">[38] </span><strong><span class="ne-text">EndoGSLAM</span></strong><strong><span class="ne-text" style="color: #DF2A3F">*</span></strong><strong><span class="ne-text">: Real-Time Dense Reconstruction and Tracking in Endoscopic Surgeries Using Gaussian Splatting（arXiv 2024）</span></strong><span class="ne-text">：提出面向内窥镜手术场景的实时 SLAM 框架，将 3D Gaussian Splatting 引入内窥镜位姿跟踪与稠密重建过程，通过高斯显式表示联合优化相机位姿与场景几何，在弱纹理、强反光和软组织形变等复杂条件下实现高质量重建与稳定跟踪，是 3DGS-SLAM 在内窥镜手术应用中的代表性工作。  </span></p></details>
### 消融实验（Ablation Study）
我们还对 Endo-2DTAM 进行了消融实验，评估了**跟踪损失函数、建图监督模态**以及**关键帧采样策略**的影响。

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767601141976-694eefd5-b878-4de3-9b23-9ec163def097.png)

**表 3： C3VD/trans t1 b 上的跟踪损失消融实验（Tracking Loss Ablation）  **

如_表 3_ 所示，当跟踪模块同时结合点到点距离、点到平面距离和颜色损失时，**面向表面法向的跟踪**能够有效稳定轨迹估计并提升重建质量。

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767601199921-c0ff22e6-c18a-4925-a4e4-70ea36feb8ff.png)

**表 4： C3VD/trans t1 b 上的建图模态消融实验（Mapping Modality Ablation）**

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767601247878-43466116-46d3-4eb4-ae59-482cd20aaf8d.png)

**图 4. 建图模态消融的表面法向比较（Surface Normal Comparison of Mapping Ablation）**

我们比较了使用不同监督模态渲染的表面法向结果。实验结果表明，同时利用**颜色、深度和法向**作为监督的组合能够获得最佳的表面法向质量。  

在_表 4_ 中，我们验证了建图_监督模态 _的最优选择，结果表明同时利用**颜色、深度和表面法向**可获得最佳性能。_图 4 _展示了在不同建图监督模态下渲染表面法向的定性结果。

<details class="lake-collapse"><summary id="ua8efaec6"><em><span class="ne-text">监督模态 </span></em><span class="ne-text">：指</span><span class="ne-text" style="color: rgb(51, 51, 51)">在建图（mapping）过程中用于监督和优化3D重建的不同类型的数据源。系统通过比较渲染结果与真实数据来计算损失函数，从而优化高斯参数。</span></summary><p id="uf22df1d4" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2026/png/45861457/1768121795910-98c08544-9b04-48d8-b54d-601f0799a316.png" width="574.3865362091534" title="" crop="0,0,1,1" id="u811d8cab" class="ne-image"></p></details>
<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2026/png/45861457/1767601305279-5ad110ea-67d7-4e7f-97ba-1bfb31f53da3.png)

 **表 5：C3VD/trans t1 b 上的关键帧采样策略与束调整消融实验（Sampling Strategy and BA Ablation）**  

如_表 5_ 所示，采用**位姿一致的关键帧采样策略**优于其他采样策略，实现了最显著的性能提升。同时，我们进一步验证了引入**束调整（Bundle Adjustment）**可带来整体性能的提升。

## CONCLUSIONS
在本文中，我们提出了 **Endo-2DTAM**，一种利用 **二维高斯溅射（2D Gaussian Splatting）** 的新型稠密 SLAM 系统，用于提升内窥镜重建效果。我们的方法有效缓解了基于 3DGS 的 SLAM 中的视角不一致问题，为外科医生提供了增强的新视角合成、视角一致的深度估计以及精确的表面法向，这些对于术中精确可视化至关重要。

通过大量实验，我们验证了 Endo-2DTAM 在手术场景几何重建上达到了最先进的性能，同时保持了计算高效的跟踪能力。然而，当前实现可能在组织快速变形的场景中面临挑战，这是后续改进的方向。未来工作中，我们将致力于降低 Endo-2DTAM 对深度的依赖，并将其集成至机器人辅助手术流程中，用于规划与导航。

