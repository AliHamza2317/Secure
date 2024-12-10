import {React, useEffect} from "react";
import { Helmet } from "react-helmet";
// Screens
import Landing from "./screens/Landing.jsx";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Login2 from "./dashboard/views/authentication/Login.js";
import Register2 from "./dashboard/views/authentication/Register.js";
// import FullLayout from './dashboard/layouts/full/FullLayout.js'; 
import ForgotPass from "./dashboard/views/authentication/forgotpass.jsx";
import Error from "./dashboard/views/authentication/Error.js";
// import Faqs from "./authenticationPages/faqs.jsx";
import Success from "./stripePayment/Success/Success.jsx";
import Cancel from "./stripePayment/Cancel/Cancel.jsx";
import Payment from "./stripePayment/payment.jsx";
import ResetPass from "./dashboard/views/authentication/resetPass.jsx";
// import Contact2 from "./components/Sections/Contact2.jsx";
import Contact2 from "./components/Sections/Contact2.jsx";
// import VerticalNavbar from "./nav/verticalNavbar.jsx";
import Layout from "./nav/layout.jsx";
// import Dashboard from "./nav/premium.jsx";
import Faqs from "./nav/faqs.jsx";
import Free from "./nav/free.jsx";
import Basic from "./nav/basic.jsx";
import Premium from "./nav/premium.jsx"
import { useNavigate, useLocation } from "react-router-dom";

const RequireAuth = ({ children }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const token = localStorage.getItem('token');

  useEffect(() => {
    if (!token) {
      navigate('/login', { replace: true, state: { from: location.pathname } });
    }
  }, [token, navigate, location]);

  return children;
};

export default function App() {
  return (
    <>
      <Helmet>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
        <link href="https://fonts.googleapis.com/css2?family=Khula:wght@400;600;800&display=swap" rel="stylesheet" />
      </Helmet>
      <BrowserRouter>
        <Routes>
          {/* <Route exact path="/dashboard" element={<FullLayout />}></Route> */}
          <Route exact path="/" element={<Landing />}></Route>
          {/* <Route exact path="/" element={<Faqs />}></Route> */}
          {/* authenticationRoutes */}
          <Route exact path="/login" element={<Login2 />}></Route>
          <Route exact path="/signup" element={<Register2 />}></Route>
          <Route exact path="/forgot" element={<ForgotPass />}></Route>
          <Route path="/reset/:token" element={<ResetPass />} />
          {/* authenticationRoutes */}
          {/* stripePaymentRoutes */}
          <Route exact path="/payment" element={<Payment />}></Route>
          <Route exact path="/successStripe" element={<Success />}></Route>
          <Route exact path="/cancelStripe" element={<Cancel />}></Route>
          {/* stripePaymentRoutes */}
          {/* errorRoute */}
          <Route exact path="*" element={<Error />}></Route> 
          {/* errorRoute */}
          <Route exact path="/contact" element={<Contact2 />}></Route>

           {/* Layout for the /nav routes */}
           <Route path="/nav/*" element={<Layout />}>
            <Route index element={
              <RequireAuth>
                <Free />
              </RequireAuth>
            } /> {/* Default component */}
            <Route path="basic" element={
              <RequireAuth>
                <Basic />
              </RequireAuth>
            } />
            <Route path="premium" element={
              <RequireAuth>
                <Premium />
              </RequireAuth>
            } />
            <Route path="faqs" element={<Faqs />} />
            {/* Add more nested routes as needed */}
          </Route>
          {/* <Route path="dashboard" element={<Recommend />} /> */}

        </Routes>
      </BrowserRouter>
    </>
  );
}
