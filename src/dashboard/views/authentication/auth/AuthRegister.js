// import React from 'react';
// import { Box, Typography, Button } from '@mui/material';
// import { Link } from 'react-router-dom';

// import CustomTextField from '../../../components/forms/theme-elements/CustomTextField';
// import { Stack } from '@mui/system';

// const AuthRegister = ({ title, subtitle, subtext }) => (
//     <>
//         {title ? (
//             <Typography fontWeight="700" variant="h2" mb={1}>
//                 {title}
//             </Typography>
//         ) : null}

//         {subtext}

//         <Box>
//             <Stack mb={3}>
//                 <Typography variant="subtitle1"
//                     fontWeight={600} component="label" htmlFor='name' mb="5px">Name</Typography>
//                 <CustomTextField id="name" variant="outlined" fullWidth />

//                 <Typography variant="subtitle1"
//                     fontWeight={600} component="label" htmlFor='email' mb="5px" mt="25px">Email Address</Typography>
//                 <CustomTextField id="email" variant="outlined" fullWidth />

//                 <Typography variant="subtitle1"
//                     fontWeight={600} component="label" htmlFor='password' mb="5px" mt="25px">Password</Typography>
//                 <CustomTextField id="password" variant="outlined" fullWidth type="password"  />
//             </Stack>
//             <Button color="primary" variant="contained" size="large" fullWidth component={Link} to="/login">
//                 Sign Up
//             </Button>
//         </Box>
//         {subtitle}
//     </>
// );

// export default AuthRegister;
// ____
// import React, { useState } from 'react';
// import { Box, Typography, Button } from '@mui/material';
// import { Link } from 'react-router-dom';
// import axios from 'axios'; // Import Axios for making HTTP requests

// import CustomTextField from '../../../components/forms/theme-elements/CustomTextField';
// import { Stack } from '@mui/system';

// const AuthRegister = ({ title, subtitle, subtext }) => {
//     const [formData, setFormData] = useState({
//         name: '',
//         email: '',
//         password: ''
//     });

//     const handleChange = (e) => {
//         setFormData({ ...formData, [e.target.id]: e.target.value });
//     };

//     const handleSubmit = async (e) => {
//         e.preventDefault();
//         try {
//             const response = await axios.post('http://localhost:3000/users/register', formData);
//             console.log(response.data); // Log response data
//             // You can redirect to another page upon successful registration if needed
//         } catch (error) {
//             console.error('Error registering user:', error);
//             // Handle error, show error message to the user, etc.
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

//             <Box component="form" onSubmit={handleSubmit}>
//                 <Stack mb={3}>
//                     <Typography variant="subtitle1" fontWeight={600} component="label" htmlFor="name" mb="5px">
//                         Name
//                     </Typography>
//                     <CustomTextField id="name" variant="outlined" fullWidth onChange={handleChange} />

//                     <Typography variant="subtitle1" fontWeight={600} component="label" htmlFor="email" mb="5px" mt="25px">
//                         Email Address
//                     </Typography>
//                     <CustomTextField id="email" variant="outlined" fullWidth onChange={handleChange} />

//                     <Typography variant="subtitle1" fontWeight={600} component="label" htmlFor="password" mb="5px" mt="25px">
//                         Password
//                     </Typography>
//                     <CustomTextField id="password" variant="outlined" fullWidth type="password" onChange={handleChange} />
//                 </Stack>
//                 <Button type="submit" color="primary" variant="contained" size="large" fullWidth>
//                     Sign Up
//                 </Button>
//             </Box>
//             {subtitle}
//         </>
//     );
// };

// export default AuthRegister;

 

// 3rd 
// import React, { useState } from 'react';
// import { Box, Typography, Button, Alert } from '@mui/material';
// import { Link } from 'react-router-dom';
// import axios from 'axios'; // Import Axios for making HTTP requests

// import CustomTextField from '../../../components/forms/theme-elements/CustomTextField';
// import { Stack } from '@mui/system';

// const AuthRegister = ({ title, subtitle, subtext }) => {
//     const [formData, setFormData] = useState({
//         name: '',
//         email: '',
//         password: ''
//     });
//     const [errorMessage, setErrorMessage] = useState('');

//     const handleChange = (e) => {
//         setFormData({ ...formData, [e.target.id]: e.target.value });
//     };

//     const handleSubmit = async (e) => {
//         e.preventDefault();
//         // Check if any field is empty
//         for (const key in formData) {
//             if (formData[key] === '') {
//                 setErrorMessage('Please fill out all fields');
//                 return;
//             }
//         }
//         try {
//             const response = await axios.post('http://localhost:3000/users/register', formData);
//             console.log(response.data); // Log response data
//             // Reset form data and error message on successful submission
//             setFormData({
//                 name: '',
//                 email: '',
//                 password: ''
//             });
//             setErrorMessage("");
//             // You can redirect to another page upon successful registration if needed
//         } catch (error) {
//             console.error('Error registering user:', error);
//             // Handle error, show error message to the user, etc.
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

//             <Box component="form" onSubmit={handleSubmit}>
//                 <Stack mb={3}>
//                     <Typography variant="subtitle1" fontWeight={600} component="label" htmlFor="name" mb="5px">
//                         Name
//                     </Typography>
//                     <CustomTextField id="name" variant="outlined" fullWidth onChange={handleChange} />

//                     <Typography variant="subtitle1" fontWeight={600} component="label" htmlFor="email" mb="5px" mt="25px">
//                         Email Address
//                     </Typography>
//                     <CustomTextField id="email" variant="outlined" fullWidth onChange={handleChange} />

//                     <Typography variant="subtitle1" fontWeight={600} component="label" htmlFor="password" mb="5px" mt="25px">
//                         Password
//                     </Typography>
//                     <CustomTextField id="password" variant="outlined" fullWidth type="password" onChange={handleChange} />
//                 </Stack>
//                 <Button type="submit" color="primary" variant="contained" size="large" fullWidth>
//                     Sign Up
//                 </Button>
//                 {errorMessage && (
//                     <Alert severity="error" mt={2}>
//                         {errorMessage}
//                     </Alert>
//                 )}
//             </Box>
//             {subtitle}
//         </>
//     );
// };

// export default AuthRegister;


// updated 
// import React, { useState } from 'react';
// import { Box, Typography, Button, Alert } from '@mui/material';
// import { Link } from 'react-router-dom';
// import axios from 'axios'; // Import Axios for making HTTP requests

// import CustomTextField from '../../../components/forms/theme-elements/CustomTextField';
// import { Stack } from '@mui/system';

// const AuthRegister = ({ title, subtitle, subtext }) => {
//     const [formData, setFormData] = useState({
//         name: '',
//         email: '',
//         password: ''
//     });
//     const [errorMessage, setErrorMessage] = useState('');

//     const handleChange = (e) => {
//         setFormData({ ...formData, [e.target.id]: e.target.value });
//     };

//     const handleSubmit = async (e) => {
//         e.preventDefault();
//         // Check if any field is empty
//         for (const key in formData) {
//             if (formData[key] === '') {
//                 setErrorMessage('Please fill out all fields');
//                 return;
//             }
//         }
//         try {
//             const response = await axios.post('http://localhost:3000/users/register', formData);
//             console.log(response.data); // Log response data
//             // Reset form data and error message on successful submission
//             setFormData({
//                 name: '',
//                 email: '',
//                 password: ''
//             });
//             setErrorMessage("");
//             // You can redirect to another page upon successful registration if needed
//         } catch (error) {
//             console.error('Error registering user:', error);
//             // Handle error, show error message to the user, etc.
//             if (error.response) {
//                 if (error.response.status === 400) {
//                     // Bad Request: Invalid credentials
//                     setErrorMessage('Invalid credentials. Please check your email format and ensure it is unique.');
//                 } else if (error.response.status === 409) {
//                     // Conflict: Email already exists
//                     setErrorMessage('Email already exists. Please choose a different email.');
//                 } else {
//                     // Other errors
//                     setErrorMessage('An error occurred while processing your request. Please try again later.');
//                 }
//             } else {
//                 setErrorMessage('An error occurred while processing your request. Please try again later.');
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

//             <Box component="form" onSubmit={handleSubmit}>
//                 <Stack mb={3}>
//                     <Typography variant="subtitle1" fontWeight={600} component="label" htmlFor="name" mb="5px">
//                         Name
//                     </Typography>
//                     <CustomTextField id="name" variant="outlined" fullWidth onChange={handleChange} />

//                     <Typography variant="subtitle1" fontWeight={600} component="label" htmlFor="email" mb="5px" mt="25px">
//                         Email Address
//                     </Typography>
//                     <CustomTextField id="email" variant="outlined" fullWidth onChange={handleChange} />

//                     <Typography variant="subtitle1" fontWeight={600} component="label" htmlFor="password" mb="5px" mt="25px">
//                         Password
//                     </Typography>
//                     <CustomTextField id="password" variant="outlined" fullWidth type="password" onChange={handleChange} />
//                 </Stack>
//                 <Button type="submit" color="primary" variant="contained" size="large" fullWidth>
//                     Sign Up
//                 </Button>
//                 {errorMessage && (
//                     <Alert severity="error" mt={2}>
//                         {errorMessage}
//                     </Alert>
//                 )}
//             </Box>
//             {subtitle}
//         </>
//     );
// };

// export default AuthRegister;
import React, { useState } from 'react';
import { Box, Typography, Button, Alert } from '@mui/material';
import { Link } from 'react-router-dom';
import axios from 'axios'; // Import Axios for making HTTP requests

import CustomTextField from '../../../components/forms/theme-elements/CustomTextField';
import { Stack } from '@mui/system';
import styled from 'styled-components'; // Import styled-components for styling
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';

const AuthRegister = ({ title, subtitle, subtext }) => {
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        password: ''
    });
    const [errorMessage, setErrorMessage] = useState('');
    const [showSuccessPopup, setShowSuccessPopup] = useState(false);

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.id]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        // Check if any field is empty
        for (const key in formData) {
            if (formData[key] === '') {
                setErrorMessage('Please fill out all fields');
                return;
            }
        }
        try {
            const response = await axios.post('https://secure-rec-backend.vercel.app/users/register', formData);
            console.log(response.data); // Log response data
            // Reset form data and error message on successful submission
            setFormData({
                name: '',
                email: '',
                password: ''
            });
            setErrorMessage('');
            setShowSuccessPopup(true); // Show success popup
        } catch (error) {
            console.error('Error registering user:', error);
            // Handle error, show error message to the user, etc.
            if (error.response) {
                if (error.response.status === 400) {
                    // Bad Request: Invalid credentials
                    setErrorMessage('Invalid credentials. Please check your email format and ensure it is unique.');
                } else if (error.response.status === 409) {
                    // Conflict: Email already exists
                    setErrorMessage('Email already exists. Please choose a different email.');
                } else {
                    // Other errors
                    setErrorMessage('An error occurred while processing your request. Please try again later.');
                }
            } else {
                setErrorMessage('An error occurred while processing your request. Please try again later.');
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

            <Box component="form" onSubmit={handleSubmit}>
                <Stack mb={3}>
                    <Typography variant="subtitle1" fontWeight={600} component="label" htmlFor="name" mb="5px">
                        Name
                    </Typography>
                    <CustomTextField id="name" variant="outlined" fullWidth onChange={handleChange} />

                    <Typography variant="subtitle1" fontWeight={600} component="label" htmlFor="email" mb="5px" mt="25px">
                        Email Address
                    </Typography>
                    <CustomTextField id="email" variant="outlined" fullWidth onChange={handleChange} />

                    <Typography variant="subtitle1" fontWeight={600} component="label" htmlFor="password" mb="5px" mt="25px">
                        Password
                    </Typography>
                    <CustomTextField id="password" variant="outlined" fullWidth type="password" onChange={handleChange} />
                </Stack>
                <StyledButton type="submit" color="primary" variant="contained" size="large" fullWidth>
                    Sign Up
                </StyledButton>
                {errorMessage && (
                    <Alert severity="error" mt={2}>
                        {errorMessage}
                    </Alert>
                )}
                {showSuccessPopup && (
                    <SuccessPopup>
                        <Alert
                            action={
                                <IconButton onClick={() => {
                                    setShowSuccessPopup(false); 
                                    window.location.href = '/login'; // Update to navigate to /login
                                }} size="small">
                                    <CloseIcon />
                                </IconButton>
                            }
                            severity="success"
                            mt={2}
                        >
                            Registration successful. Please proceed to login.
                        </Alert>
                    </SuccessPopup>
                )}
            </Box>
            {subtitle}
        </>
    );
};

const StyledButton = styled(Button)`
    margin-top: 20px;
`;

const SuccessPopup = styled.div`
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 9999;
`;

export default AuthRegister;
