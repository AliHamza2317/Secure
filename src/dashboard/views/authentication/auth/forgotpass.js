// import React, { useState } from 'react';
// import {
//     Box,
//     Typography,
//     FormGroup,
//     FormControlLabel,
//     Button,
//     Stack,
//     Checkbox,
//     Alert
// } from '@mui/material';
// import { Link } from 'react-router-dom';
// import axios from 'axios'; // Import axios for making HTTP requests

// import CustomTextField from '../../../components/forms/theme-elements/CustomTextField';
// import styled from 'styled-components'; // Import styled-components for styling
// import IconButton from '@mui/material/IconButton';
// import CloseIcon from '@mui/icons-material/Close';
// const AuthForgotPass = ({ title, subtitle, subtext }) => {
//     const [email, setEmail] = useState('');
//     const [password, setPassword] = useState('');
//     const [errorMessage, setErrorMessage] = useState('');
//     const [showSuccessPopup, setShowSuccessPopup] = useState(false);

//     const handleLogin = async () => {
//         try {
//             const response = await axios.post('http://localhost:3000/users/login', { email, password });
//             // Assuming response.data.user contains the user data returned from the backend
//             console.log('Login successful:', response.data.user);
//             setShowSuccessPopup(true); // Show success popup
//             // Redirect the user or perform any other actions after successful login
//         } catch (error) {
//             if (error.response) {
//                 if (error.response.status === 401) {
//                     setErrorMessage('Invalid email or password');
//                 } else if (error.response.status === 400) {
//                     setErrorMessage('Invalid email format');
//                 } else {
//                     setErrorMessage('Login failed');
//                 }
//             } else {
//                 setErrorMessage('Something went wrong. Please try again later.');
//             }
//         }
//     };

//     return (
//         <>
//             {title ? (
//                 <Typography fontWeight="700" variant="h2" mb={1}>
//                     {title}
//                 </Typography>
//             ) : null}

//             {subtext}

//             <Stack>
//                 <Box>
//                     <Typography variant="subtitle1"
//                         fontWeight={600} component="label" htmlFor='email' mb="5px">Email</Typography>
//                     <CustomTextField
//                         id="email"
//                         variant="outlined"
//                         fullWidth
//                         value={email}
//                         onChange={(e) => setEmail(e.target.value)}
//                     />
//                 </Box>
//             </Stack>
//             <Box>
//                 <StyledButton
//                     color="primary"
//                     variant="contained"
//                     size="large"
//                     fullWidth
//                     onClick={handleLogin} // Call handleLogin function when Sign In button is clicked
//                 >
//                     Continue. 
//                 </StyledButton>
//                 {errorMessage && (
//                     <Alert severity="error" mt={2}>
//                         {errorMessage}
//                     </Alert>
//                 )}
//                 {showSuccessPopup && (
//                     <SuccessPopup>
//                         <Alert
//                             action={
//                                 <IconButton onClick={() => {
//                                     setShowSuccessPopup(false); 
//                                     window.location.href = '/'; // Update to navigate to /login
//                                 }} size="small">
//                                     <CloseIcon />
//                                 </IconButton>
//                             }
//                             severity="success"
//                             mt={2}
//                         >
//                             Password reset email sent successfully.
//                         </Alert>
//                     </SuccessPopup>
//                 )}
//             </Box>
//             {subtitle}
//         </>
//     );
// };

// const StyledButton = styled(Button)`
//     margin-top: 20px;
// `;

// const SuccessPopup = styled.div`
//     position: fixed;
//     top: 50%;
//     left: 50%;
//     transform: translate(-50%, -50%);
//     z-index: 9999;
// `;

// export default AuthForgotPass;
import React, { useState } from 'react';
import {
    Box,
    Typography,
    Button,
    Stack,
    Alert
} from '@mui/material';
import axios from 'axios'; // Import axios for making HTTP requests
import { Link } from 'react-router-dom';

import CustomTextField from '../../../components/forms/theme-elements/CustomTextField';

const AuthForgotPass = ({ title, subtitle, subtext }) => {
    const [email, setEmail] = useState('');
    const [errorMessage, setErrorMessage] = useState('');
    const [showSuccessPopup, setShowSuccessPopup] = useState(false);

    const handleForgotPassword = async () => {
        try {
            const response = await axios.post('http://localhost:3000/users/forgot', { email });
            // Assuming response.data.message contains the success message returned from the backend
            console.log('Password reset email sent successfully:', response.data.message);
            setShowSuccessPopup(true); // Show success popup
        } catch (error) {
            if (error.response) {
                if (error.response.status === 404) {
                    setErrorMessage('User not found');
                } else {
                    setErrorMessage('Failed to send password reset email');
                }
            } else {
                setErrorMessage('Something went wrong. Please try again later.');
            }
        }
    };

    return (
        <>
            {title ? (
                <Typography fontWeight="700" variant="h2" mb={1}>
                    {title}
                </Typography>
            ) : null}

            {subtext}

            <Stack>
                <Box>
                    <Typography variant="subtitle1" fontWeight={600} component="label" htmlFor='email' mb="5px">Email</Typography>
                    <CustomTextField
                        id="email"
                        variant="outlined"
                        fullWidth
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                    />
                </Box>
            </Stack>
            <Box>
                <Button
                    color="primary"
                    variant="contained"
                    size="large"
                    fullWidth
                    onClick={handleForgotPassword} // Call handleForgotPassword function when Continue button is clicked
                    sx={{
                        marginTop: '2rem', // Adjust the value as needed
                    }}
                >
                    Continue
                </Button>
                {errorMessage && (
                    <Alert severity="error" mt={2}>
                        {errorMessage}
                    </Alert>
                )}
                {showSuccessPopup && (
                    <Alert severity="success" mt={2}>
                        Password reset email sent successfully.
                    </Alert>
                )}
                <Typography
                    component={Link}
                    to="/login" // Link to the Sign In page
                    fontWeight="500"
                    sx={{
                        textDecoration: 'none',
                        color: 'primary.main',
                        display: 'block',
                        marginTop: '1rem',
                        textAlign: 'center'
                    }}
                >
                    Back to Sign In
                </Typography>
            </Box>
            {subtitle}
        </>
    );
};

export default AuthForgotPass;
