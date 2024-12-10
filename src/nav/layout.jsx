// Layout.jsx
import React from 'react';
import { Outlet } from 'react-router-dom';
import Box from '@mui/material/Box';
import VerticalNavbar from './verticalNavbar'; // Import your navbar component

const Layout = () => {
  return (
    <Box sx={{ display: 'flex' }}>
      <VerticalNavbar />
      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
        <Outlet />
      </Box>
    </Box>
  );
};

export default Layout;
