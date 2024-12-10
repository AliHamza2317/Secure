// import React from "react";
// import styled from "styled-components";
// // Components
// import ProjectBox from "../Elements/ProjectBox";
// import FullButton from "../Buttons/FullButton";
// // Assets
// // import ProjectImg1 from "../../assets/img/projects/1.png";
// import ProjectImg1 from "../../assets/img/projects/1.jpg"; 
// import ProjectImg2 from "../../assets/img/projects/2.jpg";
// import ProjectImg3 from "../../assets/img/projects/3.jpg";
// import ProjectImg4 from "../../assets/img/projects/4.jpg";
// import ProjectImg5 from "../../assets/img/projects/5.jpg";
// import ProjectImg6 from "../../assets/img/projects/6.jpg";
// import AddImage2 from "../../assets/img/add/add2.png";
// import { useNavigate } from 'react-router-dom'; // Import useNavigate

// export default function Projects() {
//   const navigate = useNavigate(); // Initialize navigate
//   return (
//     <Wrapper id="projects">
//       <div className="whiteBg">
//         <div className="container">
//           <HeaderInfo>
//             <h1 className="font40 extraBold">Our Awesome Projects</h1>
//             <p className="font13">
//               Empowering businesses with innovative solutions for seamless growth, our awesome projects embody the 
//               <br />
//               spirit of progress and excellence in digital innovation.
//             </p>
//           </HeaderInfo>
//           <div className="row textCenter">
//             <div className="col-xs-12 col-sm-4 col-md-4 col-lg-4">
//               <ProjectBox
//                 img={ProjectImg1}
//                 title="ChatEase"
//                 text="Discover how ChatEase simplifies collaboration, boosting productivity."
//                 // action={() => alert("clicked")}
//               />
//             </div>
//             <div className="col-xs-12 col-sm-4 col-md-4 col-lg-4">
//               <ProjectBox
//                 img={ProjectImg2}
//                 title="TaskEasy"
//                 text="Explore how TaskEasy streamlines task organization for individuals and teams."
//                 // action={() => alert("clicked")}
//               />
//             </div>
//             <div className="col-xs-12 col-sm-4 col-md-4 col-lg-4">
//               <ProjectBox
//                 img={ProjectImg3}
//                 title="EduMentor"
//                 text="Discover how EduMentor supports students with tailored academic help."
//                 // action={() => alert("clicked")}
//               />
//             </div>
//           </div>
//           <div className="row textCenter">
//             <div className="col-xs-12 col-sm-4 col-md-4 col-lg-4">
//               <ProjectBox
//                 img={ProjectImg4}
//                 title="FinEase"
//                 text="Learn how FinEase empowers users to manage finances with ease and confidence."
//                 // action={() => alert("clicked")}
//               />
//             </div>
//             <div className="col-xs-12 col-sm-4 col-md-4 col-lg-4">
//               <ProjectBox
//                 img={ProjectImg5}
//                 title="HealthHub"
//                 text="Explore how HealthHub helps users take control of their well-being effortlessly."
//                 // action={() => alert("clicked")}
//               />
//             </div>
//             <div className="col-xs-12 col-sm-4 col-md-4 col-lg-4">
//               <ProjectBox
//                 img={ProjectImg6}
//                 title="TravelEase"
//                 text="Discover TravelEase's seamless booking and personalized travel recommendations."
//                 // action={() => alert("clicked")}
//               />
//             </div>
//           </div>
//           <div className="row flexCenter">
//             <div style={{ margin: "50px 0", width: "200px" }}>
//               {/* <FullButton title="Load More" action={() => alert("clicked")} /> */}
//             </div>
//           </div>
//         </div>
//       </div>
//       <div className="lightBg">
//         <div className="container">
//           <Advertising className="flexSpaceCenter">
//             <AddLeft>
//               <AddLeftInner>
//                 <ImgWrapper className="flexCenter">
//                   <img className="radius8" src={AddImage2} alt="add" />
//                 </ImgWrapper>
//               </AddLeftInner>
//             </AddLeft>
//             <AddRight>
//               <h4 className="font15 semiBold">A Few Words About SecureRec</h4>
//               <h2 className="font40 extraBold">Security Made Simple</h2>
//               <p className="font12">    
//                 At SecureRec, we streamline digital security with an intuitive platform that balances powerful protection and user-friendly design. 
//                 Together, let's create a safer digital world.
//               </p>
//               <ButtonsRow className="flexNullCenter" style={{ margin: "30px 0" }}>
//                 <div style={{ width: "190px" }}>
//                   <FullButton title="Get Started" action={() => navigate('/signup')} />
//                 </div>
//                 <div style={{ width: "190px", marginLeft: "15px" }}>
//                   <FullButton title="Contact Us" action={() => navigate('/contact')} border />
//                 </div>
//               </ButtonsRow>
//             </AddRight>
//           </Advertising>
//         </div>
//       </div>
//     </Wrapper>
//   );
// }

// const Wrapper = styled.section`
//   width: 100%;
// `;
// const HeaderInfo = styled.div`
//   @media (max-width: 860px) {
//     text-align: center;
//   }
// `;
// const Advertising = styled.div`
//   padding: 100px 0;
//   margin: 100px 0;
//   position: relative;
//   @media (max-width: 1160px) {
//     padding: 60px 0 40px 0;
//   }
//   @media (max-width: 860px) {
//     flex-direction: column;
//     padding: 0 0 30px 0;
//     margin: 80px 0 0px 0;
//   }
// `;
// const ButtonsRow = styled.div`
//   @media (max-width: 860px) {
//     justify-content: space-between;
//   }
// `;
// const AddLeft = styled.div`
//   position: relative;
//   width: 50%;
//   p {
//     max-width: 475px;
//   }
//   @media (max-width: 860px) {
//     width: 80%;
//     order: 2;
//     text-align: center;
//     h2 {
//       line-height: 3rem;
//       margin: 15px 0;
//     }
//     p {
//       margin: 0 auto;
//     }
//   }
// `;
// const AddRight = styled.div`
//   width: 50%;
//   @media (max-width: 860px) {
//     width: 80%;
//     order: 2;
//   }
// `;
// const AddLeftInner = styled.div`
//   width: 100%;
//   position: absolute;
//   top: -300px;
//   left: 0;
//   @media (max-width: 1190px) {
//     top: -250px;
//   }
//   @media (max-width: 920px) {
//     top: -200px;
//   }
//   @media (max-width: 860px) {
//     order: 1;
//     position: relative;
//     top: -60px;
//     left: 0;
//   }
// `;
// const ImgWrapper = styled.div`
//   width: 100%;
//   padding: 0 15%;
//   img {
//     width: 100%;
//     height: auto;
//   }
//   @media (max-width: 400px) {
//     padding: 0;
//   }
// `;


import React, { useState, useRef, useEffect } from "react";
import styled from "styled-components";
import { IconButton } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
// Components
import FullButton from "../Buttons/FullButton";
// Assets
import AddImage2 from "../../assets/img/add/add2.png";
import { useNavigate } from 'react-router-dom'; 
import video from './demonstration.mp4'; 

export default function Projects() {
  const navigate = useNavigate(); 
  const videoRef = useRef(null); // Create a ref for the video element
  const [play, setPlay] = useState(false);

  useEffect(() => {
    const videoElement = videoRef.current;
    const handleEnded = () => setPlay(false); // Reset play state when video ends

    if (videoElement) {
      videoElement.addEventListener('ended', handleEnded);
    }

    // Cleanup event listener on unmount
    return () => {
      if (videoElement) {
        videoElement.removeEventListener('ended', handleEnded);
      }
    };
  }, []);

  const handlePlayPause = () => {
    const videoElement = videoRef.current;
    if (play) {
      videoElement.pause(); // Pause the video
    } else {
      videoElement.play(); // Play the video
    }
    setPlay(!play); // Toggle play state
  };

  return (
    <Wrapper id="projects">
      <div className="whiteBg">
        <div className="container">
          <HeaderInfo>
          <h1 className="font40 extraBold">Explore the Demonstration of SecureRec</h1>
          <p className="font13">
            Discover how SecureRec revolutionizes digital security through our intuitive platform. This demonstration video showcases our innovative solutions designed to empower businesses, ensuring seamless protection and streamlined operations. Join us as we navigate the features that make SecureRec a leader in safeguarding your digital assets.
          </p>
          </HeaderInfo>
          <VideoWrapper>
            <video 
              ref={videoRef} // Reference the video element
              controls // Enable built-in video controls
              width="100%" 
              height="100%" 
              style={{ position: 'absolute', top: 0, left: 0 }} // Ensure video covers the container
              src={video} 
              type="video/mp4" 
            />
            <PlayButton onClick={handlePlayPause}>
              <IconButton>
                {play ? <PauseIcon sx={{ color: '#fff', fontSize: 60 }} /> : <PlayArrowIcon sx={{ color: '#fff', fontSize: 60 }} />}
              </IconButton>
            </PlayButton>
          </VideoWrapper>
        </div>
      </div>
      <div className="lightBg" style={{ marginTop: '50px', padding: '20px 0' }}>
        <div className="container">
          <Advertising className="flexSpaceCenter">
            <AddLeft>
              <AddLeftInner>
                <ImgWrapper className="flexCenter">
                  <img className="radius8" src={AddImage2} alt="add" />
                </ImgWrapper>
              </AddLeftInner>
            </AddLeft>
            <AddRight>
              <h4 className="font15 semiBold">A Few Words About SecureRec</h4>
              <h2 className="font40 extraBold">Security Made Simple</h2>
              <p className="font12">    
                At SecureRec, we streamline digital security with an intuitive platform that balances powerful protection and user-friendly design. 
                Together, let's create a safer digital world.
              </p>
              <ButtonsRow className="flexNullCenter" style={{ margin: "20px 0" }}>
                <div style={{ width: "190px" }}>
                  <FullButton title="Get Started" action={() => navigate('/signup')} />
                </div>
                <div style={{ width: "190px", marginLeft: "15px" }}>
                  <FullButton title="Contact Us" action={() => navigate('/contact')} border />
                </div>
              </ButtonsRow>
            </AddRight>
          </Advertising>
        </div>
      </div>
    </Wrapper>
  );
}

const Wrapper = styled.section`
  width: 100%;
`;

const HeaderInfo = styled.div`
  @media (max-width: 860px) {
    text-align: center;
  }
`;

const VideoWrapper = styled.div`
  width: 100%;
  padding: 0;
  position: relative;
  overflow: hidden;
  padding-top: 60%; /* 16:9 Aspect Ratio */ 
  // 56.25
  z-index: 1;
`;

const PlayButton = styled.div`
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 2;
  cursor: pointer;
`;

const Advertising = styled.div`
  padding: 50px 0;
  margin: 0;
  position: relative;
  @media (max-width: 1160px) {
    padding: 30px 0 20px 0;
  }
  @media (max-width: 860px) {
    flex-direction: column;
    padding: 0 0 20px 0;
    margin: 0;
  }
`;

const ButtonsRow = styled.div`
  @media (max-width: 860px) {
    justify-content: space-between;
  }
`;

const AddLeft = styled.div`
  position: relative;
  width: 50%;
  p {
    max-width: 475px;
  }
  @media (max-width: 860px) {
    width: 80%;
    order: 2;
    text-align: center;
    h2 {
      line-height: 3rem;
      margin: 15px 0;
    }
    p {
      margin: 0 auto;
    }
  }
`;

const AddRight = styled.div`
  width: 50%;
  @media (max-width: 860px) {
    width: 80%;
    order: 2;
  }
`;

const AddLeftInner = styled.div`
  width: 100%;
  position: relative;
  top: 0;
  left: 0;
`;

const ImgWrapper = styled.div`
  width: 100%;
  padding: 0 15%;
  img {
    width: 100%;
    height: auto;
  }
  @media (max-width: 400px) {
    padding: 0;
  }
`;