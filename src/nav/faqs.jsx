import React, { useState } from 'react';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline'; // FAQ icon
import { styled } from '@mui/material/styles';

const BackgroundCard = styled(Card)(({ theme }) => ({
  backgroundColor: '#ffffff', // Light background color for the card
  padding: theme.spacing(2),
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[2],
}));

const QuestionCardContent = styled(CardContent)(({ theme }) => ({
  borderBottom: `1px solid ${theme.palette.divider}`, // Divider between questions
  padding: theme.spacing(1, 0),
  '&:last-child': {
    borderBottom: 'none', // Remove border from the last item
  },
}));

const QuestionText = styled(Typography)(({ theme }) => ({
  fontSize: '0.9rem', // Smaller font size for the questions
  fontWeight: '500',  // Slightly bolder font weight
  color: '#333',      // Dark gray color for the text
}));

const AnswerText = styled(Typography)(({ theme }) => ({
  fontSize: '0.8rem', // Smaller font size for the answers
  color: '#666',      // Medium gray color for the answers
  marginTop: theme.spacing(1),
}));

const Faqs = () => {
  const [expandedIndex, setExpandedIndex] = useState(null);

  const handleToggle = (index) => {
    setExpandedIndex(expandedIndex === index ? null : index);
  };

  const faqs = [
    {
      "question": "What are the top features of SecureRec?",
      "answer": "SecureRec features automated analysis, intelligent recommendations, aspect-based sentiment analysis, and cost-effective pricing plans."
    },
    {
      "question": "What type of customer support is available from SecureRec?",
      "answer": "SecureRec provides email support for all customer inquiries."
    },
    {
      "question": "Who can benefit from using SecureRec?",
      "answer": "SecureRec benefits developers, startups, SMBs, and enterprises across various industries."
    },    
    {
      "question": "How is SecureRec commonly used?",
      "answer": "SecureRec is commonly used to automate the analysis of software reviews, provide recommendations for improvements, and conduct detailed sentiment analysis for enhanced usability and security."
    },
    {
      "question": "What benefits does SecureRec provide to software developers?",
      "answer": "SecureRec helps software developers enhance usability and security by identifying issues, offering actionable recommendations, and providing detailed sentiment insights."
    }
  ];

  return (
    <Box sx={{ maxWidth: 600, margin: 'auto', padding: 2, mt: 4 }}> {/* Increased top margin */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}> {/* Container for heading and icon */}
        <HelpOutlineIcon sx={{ color: '#666', mr: 1 }} /> {/* FAQ icon with a subtle color */}
        <Typography variant="h6" sx={{ flexGrow: 1, fontSize: '1rem', color: '#555' }}>
          SecureRec FAQs
        </Typography>
      </Box>
      <BackgroundCard>
        {faqs.map((faq, index) => (
          <QuestionCardContent key={index}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <QuestionText variant="h6">
                {faq.question}
              </QuestionText>
              <IconButton onClick={() => handleToggle(index)}>
                {expandedIndex === index ? <ExpandLessIcon /> : <ExpandMoreIcon />}
              </IconButton>
            </Box>
            {expandedIndex === index && (
              <AnswerText>
                {faq.answer.split('\n').map((line, i) => (
                  <span key={i}>{line}<br /></span>
                ))}
              </AnswerText>
            )}
          </QuestionCardContent>
        ))}
      </BackgroundCard>
    </Box>
  );
};

export default Faqs;
