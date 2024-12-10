import React from "react";
import styled from "styled-components";
// Components
import PricingTable from "../Elements/PricingTable";
import PricingTableFree from "../Elements/PricingTableFree";
export default function Pricing() {
  return (
    <Wrapper id="pricing">
      <div className="whiteBg">
        <div className="container">
          <HeaderInfo>
            <h1 className="font40 extraBold">Check Our Pricing</h1>
            <p className="font13">
              Explore our flexible pricing plans tailored to suit your needs. From basic to premium options, find
              <br />
              the perfect package to unlock the full potential of our services.
            </p>
          </HeaderInfo>
          <TablesWrapper className="flexSpaceNull">
            <TableBox>
              <PricingTableFree
                icon="browser"
                price="$0/mo"
                title="Free"
                text="Get started for free with basic aspect-based sentiment analysis."
                offers={[
                  { name: "Sentiment Analysis", cheked: true },
                  { name: "Security & Usability Issues Detection", cheked: false },
                  { name: "Detailed Recommendations", cheked: false },
                  // { name: "Product Offer", cheked: true },
                  // { name: "Offer", cheked: true },
                  // { name: "Product Offer #2", cheked: false },
                  // { name: "Product", cheked: false },
                  // { name: "Product Offer", cheked: false },
                ]}
                // action={() => alert("clicked")}
              />
            </TableBox>
            <TableBox>
              <PricingTable
                icon="roller"
                price="$49,99/mo"
                title="Basic"
                text="Upgrade to unlock aspect-based insights, and automated issue detection."
                offers={[
                  { name: "Sentiment Analysis", cheked: true },
                  { name: "Security & Usability Issues Detection", cheked: true },
                  { name: "Detailed Recommendations", cheked: false },
                  // { name: "Product Offer", cheked: true },
                  // { name: "Offer", cheked: true },
                  // { name: "Product Offer #2", cheked: true },
                  // { name: "Product", cheked: true },
                  // { name: "Product Offer", cheked: false },
                ]}
                action={() => alert("clicked")}
              />
            </TableBox>
            <TableBox>
              <PricingTable
                icon="monitor"
                price="$59,99/mo"
                title="Golden"
                text="Experience our premium package for aspect-based sentiment analysis, automated issue detection, and detailed recommendations tailored to your needs."
                offers={[
                  { name: "Sentiment Analysis", cheked: true },
                  { name: "Security & Usability Issues Detection", cheked: true },
                  { name: "Detailed Recommendations", cheked: true },
                  // { name: "Product", cheked: true },
                  // { name: "Product Offer", cheked: true },
                ]}
                action={() => alert("clicked")}
              />
            </TableBox>
          </TablesWrapper>
        </div>
      </div>
    </Wrapper>
  );
}

const Wrapper = styled.section`
  width: 100%;
  padding: 50px 0;
`;
const HeaderInfo = styled.div`
  margin-bottom: 50px;
  @media (max-width: 860px) {
    text-align: center;
  }
`;
const TablesWrapper = styled.div`
  @media (max-width: 860px) {
    flex-direction: column;
  }
`;
const TableBox = styled.div`
  width: 31%;
  @media (max-width: 860px) {
    width: 100%;
    max-width: 370px;
    margin: 0 auto
  }
`;




