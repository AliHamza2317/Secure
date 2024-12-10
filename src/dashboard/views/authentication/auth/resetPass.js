import React, { useState } from 'react';
import {
    Box,
    Typography,
    Button,
    Stack,
    Alert
} from '@mui/material';
import axios from 'axios'; // Import axios for making HTTP requests
import { useParams, Link } from 'react-router-dom';

import CustomTextField from '../../../components/forms/theme-elements/CustomTextField';

const AuthResetPass = () => {
    const { token } = useParams(); // Get token from URL params
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [errorMessage, setErrorMessage] = useState('');
    const [showSuccessPopup, setShowSuccessPopup] = useState(false);

    const handleResetPassword = async () => {
        try {
            const response = await axios.post('http://localhost:3000/users/reset-password', {
                token,
                newPassword: password
            });
            // Assuming response.data.message contains the success message returned from the backend
            console.log('Password reset successfully:', response.data.message);
            setShowSuccessPopup(true); // Show success popup
        } catch (error) {
            if (error.response) {
                if (error.response.status === 400) {
                    setErrorMessage('Invalid or expired token');
                } else {
                    setErrorMessage('Failed to reset password');
                }
            } else {
                setErrorMessage('Something went wrong. Please try again later.');
            }
        }
    };

    return (
        <>
            <Typography fontWeight="700" variant="h2" mb={1}>
                Reset Password
            </Typography>
            <Stack>
                <Box>
                    <Typography variant="subtitle1" fontWeight={600} component="label" htmlFor='password' mb="5px">New Password</Typography>
                    <CustomTextField
                        id="password"
                        variant="outlined"
                        fullWidth
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                    />
                </Box>
                <Box>
                    <Typography variant="subtitle1" fontWeight={600} component="label" htmlFor='confirm-password' mb="5px">Confirm Password</Typography>
                    <CustomTextField
                        id="confirm-password"
                        variant="outlined"
                        fullWidth
                        type="password"
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                    />
                </Box>
            </Stack>
            <Box>
                <Button
                    color="primary"
                    variant="contained"
                    size="large"
                    fullWidth
                    onClick={handleResetPassword} // Call handleResetPassword function when Continue button is clicked
                    sx={{
                        marginTop: '2rem', // Adjust the value as needed
                    }}
                >
                    Reset Password
                </Button>
                {errorMessage && (
                    <Alert severity="error" mt={2}>
                        {errorMessage}
                    </Alert>
                )}
                {showSuccessPopup && (
                    <Alert severity="success" mt={2}>
                        Password reset successfully.
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
        </>
    );
};

export default AuthResetPass;
