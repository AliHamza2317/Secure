import React,{useState} from "react";
import styled from "styled-components";
// Components
import FullButton from "../Buttons/FullButton";
// Assets
import RollerIcon from "../../assets/svg/Services/RollerIcon";
import MonitorIcon from "../../assets/svg/Services/MonitorIcon";
import BrowserIcon from "../../assets/svg/Services/BrowserIcon";
import PrinterIcon from "../../assets/svg/Services/PrinterIcon";
import CheckMark from "../../assets/svg/Checkmark";
import { useNavigate } from 'react-router-dom'; // Import useNavigate
// paymentCodeLib
import axios from 'axios';
import { loadStripe } from '@stripe/stripe-js';
const stripePromise = loadStripe('pk_test_51PHBYtHR3U6VPN4k5NgaaJ67b1owWftlHTXAYv9b63x9hIAvCfh3KgvlUPwoGNYAfUBQvlkqkIqMmusdVhEAeych00HZIy3Yso');
// paymentCodeLib
export default function PricingTable({ icon, price, title, text,  offers, action }) {
  // paymentCode
  const [loading, setLoading] = useState(false);
  const handleBuyClick = async ()=>{
    try {
      setLoading(true);
      const response = await axios.post('http://localhost:3000/users/create-checkout-session');
      const sessionData = response.data;
      const stripe = await stripePromise;
      const { error } = await stripe.redirectToCheckout({
        sessionId: sessionData.id,
      });

      if (error) {
        console.error('Error redirecting to checkout:', error);
        setLoading(false);
      }
    } catch (error) {
      console.error('Error creating checkout session:', error);
      setLoading(false);
    }
  }

  // paymentCode
  // const navigate = useNavigate(); 
  // const handleBuyClick = () => {
  //   navigate('/signup');
  // };
  let getIcon;

  switch (icon) {
    case "roller":
      getIcon = <RollerIcon />;
      break;
    case "monitor":
      getIcon = <MonitorIcon />;
      break;
    case "browser":
      getIcon = <BrowserIcon />;
      break;
    case "printer":
      getIcon = <PrinterIcon />;
      break;
    default:
      getIcon = <RollerIcon />;
      break;
  }

  return (
    <Wrapper className="whiteBg radius8 shadow">
      <div className="flexSpaceCenter">
        {getIcon}
        <p className="font30 extraBold">{price}</p>
      </div>
      <div style={{ margin: "30px 0" }}>
        <h4 className="font30 extraBold">{title}</h4>
        <p className="font13">{text}</p>
      </div>
      <div>
        {offers
          ? offers.map((item, index) => (
              <div className="flexNullCenter" style={{ margin: "15px 0" }} key={index}>
                <div style={{ position: "relative", top: "-1px", marginRight: "15px" }}>
                  {item.cheked ? (
                    <div style={{ minWidth: "20px" }}>
                      <CheckMark />
                    </div>
                  ) : (
                    <div style={{ minWidth: "20px" }}></div>
                  )}
                </div>
                <p className="font20 extraBold">{item.name}</p>
              </div>
            ))
          : null}
      </div>
      <div style={{ maxWidth: "120px", margin: "30px auto 0 auto" }}>
        <FullButton title="Buy" action={handleBuyClick} />
      </div>
    </Wrapper>
  );
}

const Wrapper = styled.div`
  width: 100%;
  text-align: left;
  padding: 20px 30px;
  margin-top: 30px;
`;
