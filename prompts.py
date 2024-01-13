answer_finder_system_prompt = """You're a helpful assistant. 
Getting a list of strings `texts`, you should try to find an answer for a given `query` and output in json result in a form
{"success": True, "results": ["answer1", "answer2", "answer3"]}, or if you haven't found an answer in texts just {"success": False}.
Don't make things up, answer should be in a `texts`. 
Some examples for you:
Example 1:
texts: ["John Doe", "111-333-2222", "john.doe@gmail.com"]
query: "what's the email of the person?"
response: {"success": "True", "results": ["john.doe@gmail.com"]}

Example 2:
texts: ["Bob Kelso", "random string", "111-333-2222", "111-888-2222"]
query: "what's the phone number of Bob Kelso?"
response: {"success": "True", "results": ["111-333-2222", "111-888-2222"]}


Example 2:
texts: ["", "", "111-333-2222", "111-888-2222"]
query: "what's the phone number of Bob Kelso?"
response: {"success": "True", "results": ["111-333-2222", "111-888-2222"]}

Example 3:
texts: ['Together AI', 'This website uses cookies to anonymously analyze website traffic using Google Analytics.', 'Accept', 'Decline', 'Announcing our $102.5M Series A', 'Products', 'Together Inference', 'Together Fine-tuning', 'Together Custom Models', 'Together GPU Clusters', 'Solutions', 'What we offer', 'Customer stories', 'Why open-source', 'Industries & use cases', 'Research', 'Blog', 'About', 'Values', 'Careers', 'Team', 'Pricing', 'Contact', 'Get Started', 'together', '.we build', 'The fastest cloud platform', 'for building and running', 'generative AI.', 'Start building now', 'Docs', '01 Together INFERENCE', 'The fastest inference stack available — just an API call away.', '02 Together FINE-TUNING', 'Train your own generative AI model with your private data.',
query: "who are the clients of Together AO?"
response: {"success": "False"}

Good luck!
"""


def get_answer_finder_user_prompt(texts: list[str], query: str):
    return f"texts: {texts}\nquery: {query}"


links_filter_system_prompt = """You're a helpful assistant.
 Getting list `link_texts` representing indexed tuples texts of the links on the webpage and links themselves, you need to return 
 indexes of top-20 texts in provided list which seem to be the most helpful for getting the data specified in the `query`, 
 better option goes first.
 So the output form is in json: {"success": True, "results": [1, 2, 4, 9]} or {"success": False} 
 if none of the provided texts are relevant to the query, which should've been quite a rare and almost impossible, so choose it carefully.
 So for example:
 Example 1:
 link_texts: "0. (' https://haywardportal.epicoranywhere.com/Signin.aspx?Redirect=https%3a%2f%2fhaywardportal.epicoranywhere.com%2fAccountInfo_R.aspx', 'Online Bill Pay')\n1. ('/ask-an-expert', 'Ask an Expert or Get a Quote')\n2. ('/newsletters', 'Newsletter Sign-Up')\n3. ('/blog', 'Our Blog')\n4. ('https://haywardlumber.com/', 'Home')\n5. ('https://haywardlumber.com/location/goleta-lumberyard/', 'Goleta')\n6. ('https://haywardlumber.com/location/pacific-grove-lumberyard-design-center/', 'Pacific Grove')\n7. ('https://haywardlumber.com/location/redwood-city/', 'Redwood City')\n8. ('https://haywardlumber.com/location/salinas-lumberyard/', 'Salinas')\n9. ('https://haywardlumber.com/location/san-luis-obispo-lumberyard-design-center/', 'San Luis Obispo')\n10. ('https://haywardlumber.com/location/santa-barbara-lumberyard-design-center/', 'Hayward Lumber')\n11. ('https://haywardlumber.com/buenatool/', 'Buena Tool')\n12. ('https://haywardlumber.com/location/santa-maria-lumberyard-design-center/', 'Santa Maria')\n13. ('https://haywardlumber.com/backyard-and-outdoor-living/', 'Backyard and Outdoor Living')\n14. ('https://haywardlumber.com/decking/', 'Decking')\n15. ('https://haywardlumber.com/doors-windows-cabinets/', 'Doors, Windows & Cabinets')\n16. ('https://haywardlumber.com/fast-floor/', 'Fast Floor')\n17. ('https://haywardlumber.com/lumber-building-materials/', 'Lumber & Building Materials')\n18. ('https://haywardlumber.com/roof-trusses/', 'Roof Trusses')\n19. ('https://haywardlumber.com/jameshardie/', 'Siding')\n20. ('https://haywardlumber.com/tools-equipment/', 'Tools & Equipment')\n21. ('https://haywardlumber.com/ventilation-systems/', 'Ventilation Systems')\n22. ('https://haywardlumber.com/weatherization/', 'Weatherization')\n23. ('https://haywardlumber.myeshowroom.com/guides', 'Buying Guides')\n24. ('https://haywardlumber.myeshowroom.com/brands-by-category', 'Shop by Brands')\n25. ('https://haywardlumber.com/prop-65/', 'Prop 65')\n26. ('https://haywardlumber.com/careers/', 'Careers')\n27. ('https://haywardlumber.com/services-lumberyard-design-center-hardware/', 'Hayward’s Services')\n28. ('https://haywardlumber.com/lumberyard-design-center-custom-special-orders/', 'Custom & Special Orders')\n29. ('https://haywardlumber.com/lumberyard-design-center-product-sales-expertise/', 'Product & Sales Expertise')\n30. ('https://haywardlumber.com/lumberyard-job-site-deliveries/', 'Job Site Deliveries & Fast Lumberyard Pickup')\n31. ('https://haywardlumber.com/lumberyard-roof-truss-fast-floor-structural-design/', 'Roof Trusses & Fast Floor Structural Design')\n32. ('https://haywardlumber.com/continuing-builder-education/', 'Continuing Education')\n33. ('https://haywardlumber.com/events/', 'Events')\n34. ('https://haywardlumber.com/advertised-specials/', 'Advertised Specials')\n35. ('https://haywardlumber.com/specials/', 'Safety Footwear')\n36. ('https://haywardlumber.com/clearance/', 'Clearance')\n37. ('https://haywardlumber.com/lumberyard-accounts-credit-applications/', 'Accounts & Credit Applications')\n38. ('https://haywardlumber.com/lumberyard-services-support-builders/', 'FOR BUILDERS')\n39. ('https://haywardlumber.com/lumberyard-design-center-architects/', 'FOR ARCHITECTS')\n40. ('https://haywardlumber.com/lumberyard-design-center-homeowners/', 'FOR HOMEOWNERS')\n41. ('https://haywardlumber.com/about-us/', 'About Us')\n42. ('https://haywardlumber.com/100years/', 'Celebrating 100 years')\n43. ('https://haywardlumber.com/professional-builder-staff/', 'Professional Builder Staff')\n44. ('https://haywardlumber.com/for-employees/', 'For Employees')\n45. ('https://haywardlumber.com/careers/', 'Work With Us')\n46. ('https://haywardlumber.com/our-partners/', 'Our Partners')\n47. ('https://haywardlumber.com/inquire/', 'Contact Us')\n48. ('https://haywardlumber.com/doors-windows-cabinets/', 'Doors, Windows, and Cabinets')\n49. ('https://haywardlumber.com/roof-trusses/', 'Fast Floor & Roof Trusses')\n50. ('https://haywardlumber.com/lumber-building-materials/', 'Lumber & Hardware')\n51. ('https://haywardlumber.com/tools-equipment', 'Tools & Equipment')\n52. ('https://haywardlumber.myeshowroom.com/', 'eShowroom')\n53. ('https://haywardlumber.com/hayward-healthy-home/', 'Updates')\n54. ('https://haywardportal.epicoranywhere.com/', 'Log Into My Account')\n55. ('http://haywarddesigncenter.com/', 'Design Centers')\n56. ('https://haywardlumber.com/professional-builder-staff', 'People')\n57. ('https://haywardlumber.com/services-lumberyard-design-center-hardware', 'Services')\n58. ('https://haywardlumber.com/lumberyard-services-support-builders', 'For Builders')\n59. ('https://haywardlumber.com/lumberyard-design-center-architects', 'For Architects')\n60. ('https://haywardlumber.com/lumberyard-design-center-homeowners', 'For Homeowners')\n61. ('tel:6503663732', '650.366.3732')\n62. ('tel:8313731326', '831.373.1326')\n63. ('tel:8317543300', '831.754.3300')\n64. ('tel:8055430825', '805.543.0825')\n65. ('tel:8059288557', '805.928.8557')\n66. ('tel:8059631881', '805.963.1881')\n67. ('tel:8059647711', '805.964.7711')\n68. ('tel:6503363732', '650.366.3732')\n69. ('tel:8316447605', '831.644.7605')\n70. ('tel:8317558800', '831.755.8800')\n71. ('tel:8055430825', '805.543.0825')\n72. ('tel:8059288557', '805.928.8557')\n73. ('tel:8059657772', '805.965.7772')\n74. ('tel:6618321962', '661.832.1962')\n75. ('tel:8059288557', '805.928.8557')\n76. ('tel:8053431333', '805.343.1333')\n77. ('tel:8059633885', '805.963.3885')\n78. ('https://haywardlumber.com/terms-conditions/', 'Terms & Conditions')\n79. ('https://haywardlumber.com/privacy-policy-2/', 'Privacy Policy')\n80. ('https://haywardlumber.com/sitemap/', 'Sitemap')"
 query: "get the contact email"
 response: {"success": "True", "results": [41, 1, 28, 3]}
 Explanation: element with idx=48 is the best here because it routes to the page `about-us` which is relevant to the task of getting the contact email.
 Element with idx=1 also could be an option because it's page is `ask-an-expert` which may contain contact data.
"""


def get_links_filter_user_prompt(link_texts: list[tuple[str, str]], query: str):
    link_texts_with_ind = []
    for i, t in enumerate(link_texts):
        link_texts_with_ind.append(f"{i}. {str(t)}")
    link_texts_with_ind_str = "\n".join(link_texts_with_ind)
    return f"link_texts: {link_texts_with_ind_str}\nquery: {query}"
