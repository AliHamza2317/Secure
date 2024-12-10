import React from 'react';
import { Grid, Box, Card, Typography, Stack } from '@mui/material';
import { Link } from 'react-router-dom';
import PageContainer from "../../components/container/PageContainer";
import Logo from '../../layouts/full/shared/logo/Logo';
import AuthRegister from './auth/AuthRegister';
import { CssBaseline, ThemeProvider } from '@mui/material';
import { baselightTheme } from "../../theme/DefaultColors"; 

import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";
// import "./style/flexboxgrid.min.css";
import "../../../style/flexboxgrid.min.css";
// import './style/index.css';
import '../../../style/index.css'; 

const Register2 = () => {
  const theme = baselightTheme;
  return (
  <ThemeProvider theme={theme}>
  <CssBaseline /> 
  <PageContainer title="Register" description="this is Register page">
    <Box
      sx={{
        position: 'relative',
        '&:before': {
          content: '""',
          background: 'radial-gradient(#d2f1df, #d3d7fa, #bad8f4)',
          backgroundSize: '400% 400%',
          animation: 'gradient 15s ease infinite',
          position: 'absolute',
          height: '100%',
          width: '100%',
          opacity: '0.3',
        },
      }}
    >
      <Grid container spacing={0} justifyContent="center" sx={{ height: '100vh' }}>
        <Grid
          item
          xs={12}
          sm={12}
          lg={4}
          xl={3}
          display="flex"
          justifyContent="center"
          alignItems="center"
        >
          <Card elevation={9} sx={{ p: 4, zIndex: 1, width: '100%', maxWidth: '500px' }}>
            <Box display="flex" alignItems="center" justifyContent="center">
              <Logo />
            </Box>
            <AuthRegister
              subtext={
                <Typography variant="subtitle1" textAlign="center" color="textSecondary" mb={1}>
                  Unlocking Actionable Insights from Reviews
                </Typography>
              }
              subtitle={
                <Stack direction="row" justifyContent="center" spacing={1} mt={3}>
                  <Typography color="textSecondary" variant="h6" fontWeight="400">
                    Already have an Account?
                  </Typography>
                  <Typography 
                    component={Link}
                    to="/login" // Link to the Login page
                    fontWeight="500"
                    sx={{
                      textDecoration: 'none',
                      color: 'primary.main',
                    }}
                  >
                    Sign In
                  </Typography>
                </Stack>
              }
            />
          </Card>
        </Grid>
      </Grid>
    </Box>
  </PageContainer>
  </ThemeProvider>
);
};

export default Register2;
