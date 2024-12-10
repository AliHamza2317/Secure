// import React from 'react';
// import {
//     Box,
//     Typography,
//     FormGroup,
//     FormControlLabel,
//     Button,
//     Stack,
//     Checkbox
// } from '@mui/material';
// import { Link } from 'react-router-dom';

// import CustomTextField from '../../../components/forms/theme-elements/CustomTextField';

// const AuthLogin = ({ title, subtitle, subtext }) => (
//     <>
//         {title ? (
//             <Typography fontWeight="700" variant="h2" mb={1}>
//                 {title}
//             </Typography>
//         ) : null}

//         {subtext}

//         <Stack>
//             <Box>
//                 <Typography variant="subtitle1"
//                     fontWeight={600} component="label" htmlFor='username' mb="5px">Username</Typography>
//                 <CustomTextField id="username" variant="outlined" fullWidth />
//             </Box>
//             <Box mt="25px">
//                 <Typography variant="subtitle1"
//                     fontWeight={600} component="label" htmlFor='password' mb="5px" >Password</Typography>
//                 <CustomTextField id="password" type="password" variant="outlined" fullWidth />
//             </Box>
//             <Stack justifyContent="space-between" direction="row" alignItems="center" my={2}>
//                 <FormGroup>
//                     <FormControlLabel
//                         control={<Checkbox defaultChecked />}
//                         label="Remeber this Device"
//                     />
//                 </FormGroup>
//                 <Typography
//                     component={Link}
//                     to="/"
//                     fontWeight="500"
//                     sx={{
//                         textDecoration: 'none',
//                         color: 'primary.main',
//                     }}
//                 >
//                     Forgot Password ?
//                 </Typography>
//             </Stack>
//         </Stack>
//         <Box>
//             <Button
//                 color="primary"
//                 variant="contained"
//                 size="large"
//                 fullWidth
//                 component={Link}
//                 to="/"
//                 type="submit"
//             >
//                 Sign In
//             </Button>
//         </Box>
//         {subtitle}
//     </>
// );

// export default AuthLogin;

// 2
// import React, { useState } from 'react';
// import {
//     Box,
//     Typography,
//     FormGroup,
//     FormControlLabel,
//     Button,
//     Stack,
//     Checkbox
// } from '@mui/material';
// import { Link } from 'react-router-dom';
// import axios from 'axios'; // Import axios for making HTTP requests

// import CustomTextField from '../../../components/forms/theme-elements/CustomTextField';

// const AuthLogin = ({ title, subtitle, subtext }) => {
//     const [email, setEmail] = useState('');
//     const [password, setPassword] = useState('');

//     const handleLogin = async () => {
//         try {
//             const response = await axios.post('http://localhost:3000/users/login', { email, password });
//             // Assuming response.data.user contains the user data returned from the backend
//             console.log('Login successful:', response.data.user);
//             // Redirect the user or perform any other actions after successful login
//         } catch (error) {
//             console.error('Login failed:', error.response.data.message);
//             // Handle login error (e.g., display error message to the user)
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
//                         fontWeight={600} component="label" htmlFor='username' mb="5px">Username</Typography>
//                     <CustomTextField
//                         id="username"
//                         variant="outlined"
//                         fullWidth
//                         value={email}
//                         onChange={(e) => setEmail(e.target.value)}
//                     />
//                 </Box>
//                 <Box mt="25px">
//                     <Typography variant="subtitle1"
//                         fontWeight={600} component="label" htmlFor='password' mb="5px" >Password</Typography>
//                     <CustomTextField
//                         id="password"
//                         type="password"
//                         variant="outlined"
//                         fullWidth
//                         value={password}
//                         onChange={(e) => setPassword(e.target.value)}
//                     />
//                 </Box>
//                 <Stack justifyContent="space-between" direction="row" alignItems="center" my={2}>
//                     <FormGroup>
//                         <FormControlLabel
//                             control={<Checkbox defaultChecked />}
//                             label="Remember this Device"
//                         />
//                     </FormGroup>
//                     <Typography
//                         component={Link}
//                         to="/"
//                         fontWeight="500"
//                         sx={{
//                             textDecoration: 'none',
//                             color: 'primary.main',
//                         }}
//                     >
//                         Forgot Password ?
//                     </Typography>
//                 </Stack>
//             </Stack>
//             <Box>
//                 <Button
//                     color="primary"
//                     variant="contained"
//                     size="large"
//                     fullWidth
//                     onClick={handleLogin} // Call handleLogin function when Sign In button is clicked
//                 >
//                     Sign In
//                 </Button>
//             </Box>
//             {subtitle}
//         </>
//     );
// };

// export default AuthLogin;

// updated 
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

// const AuthLogin = ({ title, subtitle, subtext }) => {
//     const [email, setEmail] = useState('');
//     const [password, setPassword] = useState('');
//     const [errorMessage, setErrorMessage] = useState('');

//     const handleLogin = async () => {
//         try {
//             const response = await axios.post('http://localhost:3000/users/login', { email, password });
//             // Assuming response.data.user contains the user data returned from the backend
//             console.log('Login successful:', response.data.user);
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
//                 <Box mt="25px">
//                     <Typography variant="subtitle1"
//                         fontWeight={600} component="label" htmlFor='password' mb="5px" >Password</Typography>
//                     <CustomTextField
//                         id="password"
//                         type="password"
//                         variant="outlined"
//                         fullWidth
//                         value={password}
//                         onChange={(e) => setPassword(e.target.value)}
//                     />
//                 </Box>
//                 <Stack justifyContent="space-between" direction="row" alignItems="center" my={2}>
//                     <FormGroup>
//                         <FormControlLabel
//                             control={<Checkbox defaultChecked />}
//                             label="Remember this Device"
//                         />
//                     </FormGroup>
//                     <Typography
//                         component={Link}
//                         to="/"
//                         fontWeight="500"
//                         sx={{
//                             textDecoration: 'none',
//                             color: 'primary.main',
//                         }}
//                     >
//                         Forgot Password ?
//                     </Typography>
//                 </Stack>
//             </Stack>
//             <Box>
//                 <Button
//                     color="primary"
//                     variant="contained"
//                     size="large"
//                     fullWidth
//                     onClick={handleLogin} // Call handleLogin function when Sign In button is clicked
//                 >
//                     Sign In
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

// export default AuthLogin;
import React, { useState } from 'react';
import {
    Box,
    Typography,
    FormGroup,
    FormControlLabel,
    Button,
    Stack,
    Checkbox,
    Alert
} from '@mui/material';
import { Link } from 'react-router-dom';
import axios from 'axios'; // Import axios for making HTTP requests
import CustomTextField from '../../../components/forms/theme-elements/CustomTextField';
import styled from 'styled-components'; // Import styled-components for styling
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
const AuthLogin = ({ title, subtitle, subtext }) => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [errorMessage, setErrorMessage] = useState('');
    const [showSuccessPopup, setShowSuccessPopup] = useState(false);

    const handleLogin = async () => {
        try {
            const response = await axios.post('https://secure-rec-backend.vercel.app/users/login', { email, password });
            
            const token = response.data.token;
            localStorage.setItem('token', token); // Store token in local storage
            
            // Assuming response.data.user contains the user data returned from the backend
            console.log('Login successful:', response.data.user);
            setShowSuccessPopup(true); // Show success popup
            // Redirect the user or perform any other actions after successful login
        } catch (error) {
            if (error.response) {
                if (error.response.status === 401) {
                    setErrorMessage('Invalid email or password');
                } else if (error.response.status === 400) {
                    setErrorMessage('Invalid email format');
                } else {
                    setErrorMessage('Login failed');
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
                    <Typography variant="subtitle1"
                        fontWeight={600} component="label" htmlFor='email' mb="5px">Email</Typography>
                    <CustomTextField
                        id="email"
                        variant="outlined"
                        fullWidth
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                    />
                </Box>
                <Box mt="25px">
                    <Typography variant="subtitle1"
                        fontWeight={600} component="label" htmlFor='password' mb="5px" >Password</Typography>
                    <CustomTextField
                        id="password"
                        type="password"
                        variant="outlined"
                        fullWidth
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                    />
                </Box>
                <Stack justifyContent="space-between" direction="row" alignItems="center" my={2}>
                    <FormGroup>
                        <FormControlLabel
                            control={<Checkbox defaultChecked />}
                            label="Remember this Device"
                        />
                    </FormGroup>
                    <Typography
                        component={Link}
                        to="/forgot"
                        fontWeight="500"
                        sx={{
                            textDecoration: 'none',
                            color: 'primary.main',
                        }}
                    >
                        Forgot Password ?
                    </Typography>
                </Stack>
            </Stack>
            <Box>
                <StyledButton
                    color="primary"
                    variant="contained"
                    size="large"
                    fullWidth
                    onClick={handleLogin} // Call handleLogin function when Sign In button is clicked
                >
                    Sign In
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
                                    window.location.href = '/nav/premium'; // Update to navigate to /login
                                }} size="small">
                                    <CloseIcon />
                                </IconButton>
                            }
                            severity="success"
                            mt={2}
                        >
                            Login successful. Proceed to your account.
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

export default AuthLogin;
