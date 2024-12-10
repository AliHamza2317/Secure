import React from 'react';
import { Link } from 'react-router-dom';
import { Grid, Box, Card, Stack, Typography } from '@mui/material';
import { CssBaseline, ThemeProvider } from '@mui/material';
import { baselightTheme } from "../../theme/DefaultColors"; 
// components
// import PageContainer from 'src/components/container/PageContainer';
import PageContainer from "../../components/container/PageContainer";
import Logo from '../../layouts/full/shared/logo/Logo';
import AuthLogin from './auth/AuthLogin';

const Login2 = () => {
  const theme = baselightTheme;
  return (
    <ThemeProvider theme={theme}>
    <CssBaseline />
    <PageContainer title="Login" description="this is Login page">
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
              <AuthLogin
                subtext={
                  <Typography variant="subtitle1" textAlign="center" color="textSecondary" mb={1}>
                    Unlocking Actionable Insights from Reviews
                  </Typography>
                }
                subtitle={
                  <Stack direction="row" spacing={1} justifyContent="center" mt={3}>
                    <Typography color="textSecondary" variant="h6" fontWeight="500">
                      New to SecureRec?
                    </Typography>
                    <Typography
                      component={Link}
                      to="/signup" // Link to the Register page
                      fontWeight="500"
                      sx={{
                        textDecoration: 'none',
                        color: 'primary.main',
                      }}
                    >
                      Create an account
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

export default Login2;
