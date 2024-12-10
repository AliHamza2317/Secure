import React, { useState } from 'react';
import axios from 'axios';
import { loadStripe } from '@stripe/stripe-js';
const stripePromise = loadStripe('pk_test_51PHBYtHR3U6VPN4k5NgaaJ67b1owWftlHTXAYv9b63x9hIAvCfh3KgvlUPwoGNYAfUBQvlkqkIqMmusdVhEAeych00HZIy3Yso');

const Payment = () => {
  const [loading, setLoading] = useState(false);

  const handleClick = async () => {
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
  };

  return (
    <button onClick={handleClick} disabled={loading}>
      {loading ? 'Loading...' : 'Checkout'}
    </button>
  );
};

export default Payment;
