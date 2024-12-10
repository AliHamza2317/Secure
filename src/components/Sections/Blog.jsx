import React from "react";
import styled from "styled-components";
// Components
import BlogBox from "../Elements/BlogBox";
// import FullButton from "../Buttons/FullButton";
import TestimonialSlider from "../Elements/TestimonialSlider";
import Faqs from "../../nav/faqs";

export default function Blog() {
  return (
    <Wrapper id="blog">
      <div className="whiteBg">
        <div className="container">
          <HeaderInfo>
            <h1 className="font40 extraBold">Faqs</h1>
            <p className="font13">
              Find quick answers to your questions about SecureRec. Get clear, concise information on our features,
              <br />
              services, and support.
            </p>
          </HeaderInfo>
          <Faqs />
          {/* <TestimonialSlider /> */}
        </div>
      </div>
    </Wrapper>
  );
}

const Wrapper = styled.section`
  width: 100%;
  padding-top: 20px;
`;

const HeaderInfo = styled.div`
  margin-bottom: 30px;
  @media (max-width: 860px) {
    text-align: center;
  }
`;
