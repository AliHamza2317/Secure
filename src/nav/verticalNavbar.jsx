import React, { useState } from 'react';
import Box from '@mui/material/Box';
import Drawer from '@mui/material/Drawer';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import { styled } from '@mui/material/styles';
import DashboardIcon from '@mui/icons-material/Dashboard';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import ExitToAppIcon from '@mui/icons-material/ExitToApp';
import MenuIcon from '@mui/icons-material/Menu';
import IconButton from '@mui/material/IconButton';
import Hidden from '@mui/material/Hidden';
import logo from "../assets/img/logo.png";
import { Link, useNavigate } from 'react-router-dom'; // Import Link for navigation

const drawerWidth = 240;

const CustomDrawer = styled(Drawer)(({ theme }) => ({
  '& .MuiDrawer-paper': {
    width: drawerWidth,
    boxSizing: 'border-box',
    backgroundColor: '#1A1A1A',
    color: '#FFFFFF',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between', // Distribute space between top and bottom
    height: '100%',
  },
}));

const CustomListItem = styled(ListItem)(({ theme }) => ({
  '&:hover': {
    backgroundColor: 'transparent',
    transform: 'translateX(5px)',
  },
  transition: 'transform 0.2s ease-in-out',
}));

const CustomListItemText = styled(ListItemText)(({ theme }) => ({
  '& .MuiTypography-root': {
    color: '#FFFFFF',
  },
  '&:hover .MuiTypography-root': {
    color: '#FFFFFF',
  },
}));

const VerticalNavbar = () => {
  const [mobileOpen, setMobileOpen] = useState(false);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const navigate = useNavigate();
  const handleLogout = ()=>{
    localStorage.removeItem('token'); 
    navigate('/');
  }

  const drawerContent = (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mt: 2 }}>
        {/* <img src={logo} alt="Logo" style={{ width: '70px', height: 'auto' }} /> */}
        <Link to="/" style={{ display: 'block', width: '70px', height: 'auto' }}>
          <img src={logo} alt="Logo" style={{ width: '100%', height: 'auto' }} />
        </Link>
      </Box>
      <List sx={{ flexGrow: 1 }}>
        <CustomListItem button component={Link} to="/nav/premium">
          <ListItemIcon>
            <DashboardIcon sx={{ color: '#FFFFFF' }} />
          </ListItemIcon>
          <CustomListItemText primary="Dashboard" />
        </CustomListItem>
        <CustomListItem button component={Link} to="/nav/faqs">
          <ListItemIcon>
            <HelpOutlineIcon sx={{ color: '#FFFFFF' }} />
          </ListItemIcon>
          <CustomListItemText primary="FAQs" />
        </CustomListItem>
      </List>
      <List>
        <CustomListItem button component={Link} to="/" onClick={handleLogout}>
          <ListItemIcon>
            <ExitToAppIcon sx={{ color: '#FFFFFF' }} />
          </ListItemIcon>
          <CustomListItemText primary="Logout" />
        </CustomListItem>
      </List>
    </div>
  );

  return (
    <Box sx={{ display: 'flex', position: 'relative' }}>
      <Hidden mdUp>
        <IconButton
          color={mobileOpen ? 'default' : 'inherit'}
          aria-label="open drawer"
          edge="start"
          onClick={handleDrawerToggle}
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            margin: 1,
            zIndex: 1300, // Ensure it's above other elements
            color: mobileOpen ? '#FFFFFF' : '#000000', // Change color based on drawer state
          }}
        >
          <MenuIcon />
        </IconButton>
        <CustomDrawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better open performance on mobile.
          }}
        >
          {drawerContent}
        </CustomDrawer>
      </Hidden>
      <Hidden mdDown>
        <CustomDrawer variant="permanent" open>
          {drawerContent}
        </CustomDrawer>
      </Hidden>
    </Box>
  );
};

export default VerticalNavbar;
