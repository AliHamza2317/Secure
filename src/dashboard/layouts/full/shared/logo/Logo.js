import { Link } from 'react-router-dom';
// import { ReactComponent as LogoDark } from 'src/assets/images/logos/dark-logo.svg';
// import { ReactComponent as LogoDark } from '../../../../assets/images/logos/dark-logo.svg'; 
import { styled } from '@mui/material';
import LogoImage from '../../../../assets/images/logos/logo.png'; // Import the PNG image file
const LinkStyled = styled(Link)(() => ({
  height: '70px',
  width: '180px',
  overflow: 'hidden',
  // display: 'block',
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center', // Center the content vertically
  textDecoration: 'none'
}));

const Logo = () => {
  return (
    <LinkStyled to="/">
      {/* <LogoDark height={70} /> */}
      <img src={LogoImage} alt="Logo" height={70} width={70} /> {/* Use the imported PNG image */}
    </LinkStyled>
  )
};

export default Logo;
