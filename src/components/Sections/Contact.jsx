import React, { useState } from "react";
import styled from "styled-components";
import axios from "axios";
import Alert from '@mui/material/Alert';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
// Assets
import ContactImg1 from "../../assets/img/contact-1.png";
import ContactImg2 from "../../assets/img/contact-2.png";
import ContactImg3 from "../../assets/img/contact-3.png";

export default function Contact() {
  const [formData, setFormData] = useState({
    firstName: "",
    email: "",
    subject: "",
    message: ""
  });
  const [errorMessage, setErrorMessage] = useState("");
  const [showSuccessPopup, setShowSuccessPopup] = useState(false);
  const [showSpinner, setShowSpinner] = useState(false);

  // Email validation function
  const validateEmail = (email) => {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(String(email).toLowerCase());
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setShowSpinner(true); // Show spinner when request starts

    // Check if any field is empty
    for (const key in formData) {
      if (formData[key] === "") {
        setErrorMessage("Please fill out all fields");
        setShowSpinner(false); // Hide spinner if validation fails
        return;
      }
    }

     // Validate email format
    if (!validateEmail(formData.email)) {
      setErrorMessage("Please enter a valid email address");
      setShowSpinner(false); // Hide spinner if email validation fails
      return;
    }

    try {
      const response = await axios.post("https://secure-rec-backend.vercel.app/api/submit", formData);
      console.log(response.data.message);
      // Reset form data and error message on successful submission
      setFormData({
        firstName: "",
        email: "",
        subject: "",
        message: ""
      });
      setErrorMessage("");
      setShowSpinner(false); // Hide spinner
      setShowSuccessPopup(true); // Show the success popup
    } catch (error) {
      console.error(error);
      // Handle error
      setShowSpinner(false); // Hide spinner if request fails
    }
  };

  return (
    <Wrapper id="contact">
      <div className="lightBg">
        <div className="container">
          <HeaderInfo>
            <h1 className="font40 extraBold">Contact us</h1>
            <p className="font13">
              Please feel free to reach out to us with any questions you may have!
            </p>
          </HeaderInfo>
          <div className="row" style={{ paddingBottom: "30px" }}>
            <div className="col-xs-12 col-sm-12 col-md-6 col-lg-6">
              <Form onSubmit={handleSubmit}>
                <label className="font13">First name:</label>
                <input
                  type="text"
                  name="firstName"
                  className="font20 extraBold"
                  value={formData.firstName}
                  onChange={handleChange}
                />
                <label className="font13">Email:</label>
                <input
                  type="text"
                  name="email"
                  className="font20 extraBold"
                  value={formData.email}
                  onChange={handleChange}
                />
                <label className="font13">Subject:</label>
                <input
                  type="text"
                  name="subject"
                  className="font20 extraBold"
                  value={formData.subject}
                  onChange={handleChange}
                />
                <textarea
                  rows="4"
                  cols="50"
                  name="message"
                  className="font20 extraBold"
                  value={formData.message}
                  onChange={handleChange}
                />
                <SubmitButton type="submit" className="pointer animate radius8">
                  Send Message
                </SubmitButton>
                {errorMessage && (
                  <Alert severity="error">
                    {errorMessage}
                  </Alert>
                )}
                {showSpinner && //<Spinner color="secondary" />} {/* Show spinner conditionally */}
                <Spinner viewBox="0 0 50 50">
    <circle
      className="path"
      cx="25"
      cy="25"
      r="20"
      fill="none"
      strokeWidth="4"
    />
                </Spinner> }
              </Form>
              {showSuccessPopup && (
                <SuccessPopup>
                  <Alert
                    action={
                      <IconButton
                        aria-label="close"
                        color="inherit"
                        size="small"
                        onClick={() => {
                          setShowSuccessPopup(false);
                        }}
                      >
                        <CloseIcon fontSize="inherit" />
                      </IconButton>
                    }
                    sx={{ mb: 2 }}
                  >
                    Message has been received successfully. We will get back to you soon!
                  </Alert>
                </SuccessPopup>
              )}
            </div>
            <div className="col-xs-12 col-sm-12 col-md-6 col-lg-6 flex">
              <div style={{ width: "50%" }} className="flexNullCenter flexColumn">
                <ContactImgBox>
                  <img src={ContactImg1} alt="office" className="radius6" />
                </ContactImgBox>
                <ContactImgBox>
                  <img src={ContactImg2} alt="office" className="radius6" />
                </ContactImgBox>
              </div>
              <div style={{ width: "50%" }}>
                <div style={{ marginTop: "100px" }}>
                  <img src={ContactImg3} alt="office" className="radius6" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Wrapper>
  );
}

const Wrapper = styled.section`
  width: 100%;
`;
const HeaderInfo = styled.div`
  padding: 70px 0 30px 0;
  @media (max-width: 860px) {
    text-align: center;
  }
`;
const Form = styled.form`
  padding: 70px 0 30px 0;
  input,
  textarea {
    width: 100%;
    background-color: transparent;
    border: 0px;
    outline: none;
    box-shadow: none;
    border-bottom: 1px solid #707070;
    height: 30px;
    margin-bottom: 30px;
  }
  textarea {
    min-height: 100px;
  }
  @media (max-width: 860px) {
    padding: 30px 0;
  }
`;
const SubmitButton = styled.button`
  border: 1px solid #7620ff;
  background-color: #7620ff;
  width: 100%;
  padding: 15px;
  outline: none;
  color: #fff;
  :hover {
    background-color: #580cd2;
    border: 1px solid #7620ff;
    color: #fff;
  }
  @media (max-width: 991px) {
    margin: 0 auto;
  }
`;
const ContactImgBox = styled.div`
  max-width: 180px; 
  align-self: flex-end; 
  margin: 10px 30px 10px 0;
`;
const Spinner = styled.svg`
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 9999;
  animation: rotate 2s linear infinite;
  margin: -25px 0 0 -25px;
  width: 50px;
  height: 50px;

  & .path {
    stroke: #5652BF;
    stroke-linecap: round;
    animation: dash 1.5s ease-in-out infinite;
  }

  @keyframes rotate {
    100% {
      transform: rotate(360deg);
    }
  }
  @keyframes dash {
    0% {
      stroke-dasharray: 1, 150;
      stroke-dashoffset: 0;
    }
    50% {
      stroke-dasharray: 90, 150;
      stroke-dashoffset: -35;
    }
    100% {
      stroke-dasharray: 90, 150;
      stroke-dashoffset: -124;
    }
  }
`;
const SuccessPopup = styled.div`
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 9999;
`;